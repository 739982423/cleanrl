import math
import torch
import joblib
import numpy as np
import gym
import csv
from gym import spaces, logger
from gym.utils import seeding
from typing import Optional, Union
from sklearn import tree

GPU_NUMS = 2
MODEL_NUMS = 4
candidate_batch = [4,8,16]
candidate_gpu_resource = [i * 10 for i in range(1, 11)]

global_kernel_data = dict()
# 读入的数据行每列的含义：Model,Batch,GPU Resource,L1RL2,L1WL2,L2RD,L2WD,L2Hit,L2Miss,Latency（与summary文件一致）
with open("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\\summary3.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line < 1:
            line += 1
            continue
        model_name = row[0]
        batch = row[1]
        gpu_resource = row[2]
        key = model_name + "_b" + batch + "_g" + gpu_resource
        global_kernel_data[key] = row
        # L1RL2 = row[3]
        # L1WL2 = row[4]
        # L2RD = row[5]
        # L2WD = row[6]
        # L2Hit = row[7]
        # L2Miss = row[8]
        # latency = row[9]


class ModelPlace():
    def __init__(self, model_name, batchsize, gpu_resource, gpu_idx):
        self.model_name = model_name
        self.b = batchsize
        self.gpu_resource = gpu_resource
        self.gpu_idx = gpu_idx
        self.nomalized_L1RL2 = None
        self.nomalized_L1WL2 = None
        self.nomalized_L2RD = None
        self.nomalized_L2WD = None
        self.nomalized_L2Hit = None
        self.nomalized_L2Miss = None

        self.latency = None
        self.inter_latency = None
        self.interference = None
        self.throughput = None
        self.getKernelMessage()    

    def getKernelMessage(self):
        key = self.model_name + "_b" + str(self.b) + "_g" + str(self.gpu_resource)
        row = global_kernel_data[key]
        L1RL2 = float(row[3])
        L1WL2 = float(row[4])
        L2RD = float(row[5])
        L2WD = float(row[6])
        L2Hit = float(row[7])
        L2Miss = float(row[8])
        latency = float(row[9])

        self.nomalized_L1RL2 = L1RL2 / latency
        self.nomalized_L1WL2 = L1WL2 / latency
        self.nomalized_L2RD = L2RD / latency
        self.nomalized_L2WD = L2WD / latency
        self.nomalized_L2Hit = L2Hit / latency
        self.nomalized_L2Miss = L2Miss / latency
        self.L2_hit_rate = L2Hit / (L2Miss + L2Hit)
        self.latency = latency

    def showDetails(self):
        print("--------cur place details--------")
        print(self.model_name)
        print("placing gpu:", self.gpu_idx)
        print("batch:", self.b)
        print("gpu resource:", self.gpu_resource)
        print("latency", self.latency)
        print("inter_latency", self.inter_latency)
        print("interference", self.interference)
        print("throughput", self.throughput)

class GPU():
    def __init__(self, idx):
        self.idx = idx
        self.places = []

    def addPlaces(self, place):
        self.places.append(place)

    def showPlaces(self):
        for place in self.places:
            place.showDetails()
    
    def deletePlaces(self):
        self.places = []

    def checkGPUResource(self):
        cur_gpu_resource = 100
        for place in self.places:
            cur_gpu_resource -= place.gpu_resource
        return cur_gpu_resource >= 0

class RequestGroup():
    def __init__(self, model_name, request_nums, create_time):
        self.model_name = model_name
        self.cur_requset_nums = request_nums
        self.create_time = create_time
    
    def showDetails(self):
        print("--- request group ---")
        print("model name", self.model_name)
        print("remaining request:", self.cur_requset_nums)
        print("create time:", self.create_time)

class GPUEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self):
        # action space: GPU_NUMS * len(candidate_batch) * len(candidate_gpu_resource)
        self.action_length = GPU_NUMS * len(candidate_batch) * len(candidate_gpu_resource)
        self.action_space = spaces.MultiDiscrete([self.action_length, self.action_length, self.action_length, self.action_length])
        # print("self.action multi shape", self.action_space.shape)
        # observation space: （模型当前时刻请求数 + 模型当前时刻BUFFER队列长度 + 模型在GPU1是否缓存 + GPU2是否缓存 +...+ GPUN是否缓存）* 模型数量
        self.observation_length = (2 + GPU_NUMS) * MODEL_NUMS

        self.observation_space = spaces.MultiDiscrete([800, 800, 800, 800,
                                                       500, 500, 500, 500,
                                                       2, 2, 2, 2, 
                                                       2, 2, 2, 2], dtype=np.float32)
        self.model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']                                              
        self.cnt = 0
        self.GPUS = [GPU(i) for i in range(GPU_NUMS)]
        self.CART_dtr = joblib.load("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\\drl\\cart.pkl") # 读取

        self.buffers = dict()
        self.buffers['resnet50'] = []
        self.buffers['vgg19'] = []
        self.buffers['densenet201'] = []
        self.buffers['mobilenet'] = []
        
        self.input_streams = dict()
        self.input_streams['resnet50'] = [i for i in range(100, 120)]
        self.input_streams['vgg19'] = [i for i in range(100, 120)]
        self.input_streams['densenet201'] = [i for i in range(100, 120)]
        self.input_streams['mobilenet'] = [i for i in range(100, 120)]
        
        self.discard_request_nums = dict()
        self.discard_request_nums['resnet50'] = [0 for _ in range(20)]
        self.discard_request_nums['vgg19'] = [0 for _ in range(20)]
        self.discard_request_nums['densenet201'] = [0 for _ in range(20)]
        self.discard_request_nums['mobilenet'] = [0 for _ in range(20)]

        self.time_scale = 1 # 代表1s
        self.time = 0       # 表示执行过程的第几个time scale
        self.request_retention_time_scale = 3 # 表示请求经过多少个time scale之后没被处理的话就丢弃了

    def showBufferDetails(self):
        for model_name in self.model_names:
            cur_buffer = self.buffers[model_name]
            print("---------- cur buffer details ----------")
            print("model name:", model_name)
            for request_group in cur_buffer:
                request_group.showDetails()

    def step(self, action): 
        # 执行该函数输出所有GPU上的所有place信息
        def showAllPlaceDetails(self):
            for GPU in self.GPUS:
                GPU.showPlaces()

        # 执行该函数则检查当前所有模型buffer中是否有过期请求，如果有过期请求则删除它们并记录SLO
        def checkExpiredRequestInBuffer(self):
            for model_name, buffer in self.buffers.items():
                new_buffer = []
                for group in buffer:
                    # 如果当前时间-请求到达时间>设置的保留时长，丢弃该group
                    if self.time - group.create_time > self.request_retention_time_scale:
                        # 丢弃前对丢弃的请求进行记录：t=group.create_time时刻，model_name的cur_requset_nums个请求没有被处理，触发SLO
                        self.discard_request_nums[group.model_name][group.create_time] += group.cur_requset_nums
                        continue
                    else:
                        new_buffer.append(group)
                buffer = new_buffer
        
        # 执行该函数则根据当前action在各个GPU上放置places
        def placeAccordingToActions(self, action, model_names):
            if len(action) != MODEL_NUMS or len(model_names) != MODEL_NUMS:
                print("每时刻动作数量与模型种类数不符~")
                return
            for i in range(len(action)):
                cur_a = action[i]
                gpu_idx = cur_a // (len(candidate_batch) * len(candidate_gpu_resource))
                cur_a -= gpu_idx * (len(candidate_batch) * len(candidate_gpu_resource))
                batch_idx = cur_a // len(candidate_gpu_resource)
                cur_a -= batch_idx * len(candidate_gpu_resource)
                gpu_resource_idx = cur_a
                new_model_place = ModelPlace(model_names[i], candidate_batch[batch_idx], 
                                             candidate_gpu_resource[gpu_resource_idx], gpu_idx)
                self.GPUS[gpu_idx].addPlaces(new_model_place)

        # 执行该函数则检查当前action表示的动作是否可行（GPU资源量大于100则不可行），需要先执行placeAccordingToActions，放置后再检查
        def checkActionFeasibility(self):
            for GPU in self.GPUS:
                if not GPU.checkGPUResource():
                    return False
            return True

        # 执行该函数则计算各个GPU上的每个place的推理时延增长量和实际推理时延，并将结果保存在place里
        def getRealThroughput(self):
            # 检查每个GPU上的place，计算每个place的输出
            for GPU in self.GPUS:
                total_L1RL2 = 0
                total_L1WL2 = 0
                total_L2RD = 0
                total_L2WD = 0
                total_L2Hit = 0
                total_L2Miss = 0
                # 首先计算总的各项数据
                for place in GPU.places:
                    total_L1RL2 += place.nomalized_L1RL2
                    total_L1WL2 += place.nomalized_L1WL2
                    total_L2RD += place.nomalized_L2RD
                    total_L2WD += place.nomalized_L2WD
                    total_L2Hit += place.nomalized_L2Hit
                    total_L2Miss += place.nomalized_L2Miss
                # 再根据每个place，计算该place的时延增长情况
                for place in GPU.places:
                    # 获取13个CART输入数据
                    cur_place_L1RL2 = place.nomalized_L1RL2
                    cur_place_L1WL2 = place.nomalized_L1WL2
                    cur_place_L2RD = place.nomalized_L2RD
                    cur_place_L2WD = place.nomalized_L2WD
                    cur_place_L2Hit = place.nomalized_L2Hit
                    cur_place_L2Miss = place.nomalized_L2Miss
                    cur_place_hit_rate = place.L2_hit_rate

                    co_located_L1RL2 = total_L1RL2 - cur_place_L1RL2
                    co_located_L1WL2 = total_L1WL2 - cur_place_L1WL2
                    co_located_L2RD = total_L2RD - cur_place_L2RD
                    co_located_L2WD = total_L2WD - cur_place_L2WD
                    co_located_L2Hit = total_L2Hit - cur_place_L2Hit
                    co_located_L2Miss = total_L2Miss - cur_place_L2Miss

                    cart_input = [cur_place_L1RL2, cur_place_L1WL2, cur_place_L2RD, cur_place_L2WD, cur_place_L2Hit, cur_place_L2Miss, cur_place_hit_rate,
                                  co_located_L1RL2, co_located_L1WL2, co_located_L2RD, co_located_L2WD, co_located_L2Hit, co_located_L2Miss]

                    cur_place_interference = self.CART_dtr.predict([cart_input])
                    place.interference = cur_place_interference
                    place.inter_latency = place.latency * (1 + cur_place_interference)

                    # 这里的吞吐量指的是一个timescale内的实际推理数量，目前timescale是1s，所以1000*1=1000
                    # 如果timescale改为0.5s，则这里的吞吐量也会减半，因为0.5s能吞吐的数量就是1s的一半
                    place.throughput = int((1000 * self.time_scale) / place.inter_latency * place.b)
        
        # 执行该函数则将输入buffer中的请求数减少take_nums个，如果buffer中没有这么多没处理完的请求，则清空buffer
        def takeRequsetOutOfBuffer(self, buffer, take_nums):
            i = 0
            buffer_inference = 0
            for idx, group in enumerate(buffer):
                if take_nums > 0:
                    if group.cur_requset_nums > take_nums:
                        group.cur_requset_nums -= take_nums
                        buffer_inference += take_nums
                        take_nums = 0
                    else:
                        take_nums -= group.cur_requset_nums
                        buffer_inference += group.cur_requset_nums
                    i = idx
            buffer = buffer[i:]
            return buffer_inference

        # 执行该函数则检查各个模型的吞吐量是否达标，是否需要buffer增减
        def checkThroughput(self, model_names):
            # 定义一个最后要最小化的变量t，t表示当前时刻所有GPU上所有place的执行时间之和
            t = 0
            throughput = dict()
            throughput['resnet50'] = 0
            throughput['vgg19'] = 0
            throughput['densenet201'] = 0
            throughput['mobilenet'] = 0
            for GPU in self.GPUS:
                for place in GPU.places:
                    model_name = place.model_name
                    throughput[model_name] += place.throughput
            for model_name in model_names:
                model_request_nums = self.input_streams[model_name][self.time]
                model_throughput = throughput[model_name]
                # 接下来分几种情况：
                # 吞吐量大于等于请求到达率，则说明当前时刻请求可完全处理，且需要查看buffer中是否有未完成的请求
                # 吞吐量小于请求到达率，当前时刻请求不能完全处理，存入buffer不能处理的部分

                # 如果请求到达率大于吞吐量，则要存入一部分进buffer
                if model_throughput < model_request_nums:
                    remaining_request_nums = model_request_nums - model_throughput
                    # 创建一个新的request_group对象，存储剩余模型名、请求数量、请求时刻
                    new_request_group = RequestGroup(model_name, remaining_request_nums, self.time)
                    # 存入buffer
                    self.buffers[model_name].append(new_request_group)
                    # 按照目前action，当前时间间隔内全部时间都用来推理该模型还推理不完所有请求呢，因此t增加一个timescale的大小
                    t += self.time_scale
                # 如果请求到达率和吞吐量相等，则无事发生，buffer不用任何变化
                elif model_throughput == model_request_nums:
                    # 当前时间间隔内全部时间都用来推理正好推理完，因此t增加一个timescale的大小
                    continue
                # 如果请求到达率小于buffer，则可以查看buffer中是否有未完成请求，可以处理这部分请求
                elif model_throughput > model_request_nums:
                    # 处理完本时刻模型请求后剩余的吞吐量数量
                    remaining_throughput_nums = model_throughput - model_request_nums
                    # 查看buffer中是否有未完成请求，如果有
                    buffer_inf = 0
                    if len(self.buffers[model_name]) >= 1:
                        # 使用一个函数处理剩余吞吐量和buffer的关系，返回实际推理的数量（是一个小于等于remaining_throughput_nums的值）
                        buffer_inf = takeRequsetOutOfBuffer(self.buffers[model_name], remaining_throughput_nums)
                        # 根据实际推理数量计算推理时长
                        # （时间间隔 / 吞吐量 = 推理一次的时间） × （本次时间间隔内的推理数量+buffer中取出的请求推理数量）
                    t += (self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                print(t)
        # ---------------- 函数执行部分 -----------------
        checkExpiredRequestInBuffer(self)
        placeAccordingToActions(self, action, self.model_names)
        checkActionFeasibility(self)
        getRealThroughput(self)
        showAllPlaceDetails(self)
        checkThroughput(self, self.model_names)
        self.showBufferDetails()

        # print("step: action:", action)
        # reward = 1
        # self.cnt += 1
        # done = False
        # if self.cnt >= 10:
        #     done = True
        # next_state = torch.rand(16)
        # return np.array(next_state, dtype=np.float32), reward, done, {}

    def reset(self):
        # new_state = np.ones(16)
        # print("reset: state:", new_state)
        # return np.array(new_state, dtype=np.float32)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(16,))
        self.steps_beyond_done = None
        # print("reset: state:", self.state)
        return np.array(self.state, dtype=np.float32)

if __name__ == "__main__":
    env = GPUEnv()
    action = [10, 20, 30, 40]
    env.step(action)
