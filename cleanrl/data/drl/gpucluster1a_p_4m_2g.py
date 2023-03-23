import collections
import math
import torch
import joblib
import numpy as np
import gym
import csv
import os
from gym import spaces, logger
from gym.utils import seeding
from typing import Optional, Union
from sklearn import tree
from itertools import combinations
import random

TWEET_INPUT = True
GPU_NUMS = 2
MODEL_NUMS = 4
candidate_batch = [4,8,16]
candidate_gpu_resource = [i * 10 for i in range(1, 11)]

global_kernel_data = collections.defaultdict(list)


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
        if batchsize not in [4,8,16]:
            if batchsize <= 4:
                self.b = 4
            elif batchsize <= 8:
                self.b = 8
            else:
                self.b = 16
        else:
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
        key = None
        if self.b not in [4,8,16]:
            if self.b <= 4:
                key = self.model_name + "_b4" + "_g" + str(self.gpu_resource)
            elif self.b <= 8:
                key = self.model_name + "_b8" + "_g" + str(self.gpu_resource)
            else:
                key = self.model_name + "_b16" + "_g" + str(self.gpu_resource)
        else:
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
        for idx, place in enumerate(self.places):
            cur_gpu_resource -= place.gpu_resource
            # print("place{}, gpu_r={}".format(idx, place.gpu_resource))
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

class GPUEnv1a_p_4m_2g(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    # 重置buffer状态
    def defineBuffers(self):
        self.buffers = collections.defaultdict(list)
        self.buffers['resnet50'] = []
        self.buffers['vgg19'] = []
        self.buffers['densenet201'] = []
        self.buffers['mobilenet'] = []
    
    # 重置输入流状态
    def defineInputStream(self):
        if not TWEET_INPUT:
            self.lambda_base = collections.defaultdict(int)
            self.lambda_base['resnet50'] = 100 + self.input_ascend
            self.lambda_base['vgg19'] = 90 + self.input_ascend
            self.lambda_base['densenet201'] = 80 + self.input_ascend
            self.lambda_base['mobilenet'] = 180 + self.input_ascend
            self.lambda_base['random_length'] = 10                  # 特意将random_length写入字典是为了传回ppo1a，在tensorboard中记录本次训练的参数
            self.random_length = self.lambda_base['random_length']

            lambda_resnet50 = random.randint(self.lambda_base['resnet50'], self.lambda_base['resnet50'] + self.random_length)
            lambda_vgg19 = random.randint(self.lambda_base['vgg19'], self.lambda_base['vgg19'] + self.random_length)
            lambda_densenet201 = random.randint(self.lambda_base['densenet201'], self.lambda_base['densenet201'] + self.random_length)
            lambda_mobilenet = random.randint(self.lambda_base['mobilenet'], self.lambda_base['mobilenet'] + self.random_length)

            self.input_streams = collections.defaultdict(list)
            self.input_streams['resnet50'] = np.random.poisson(lambda_resnet50, self.time_duration) * 2
            self.input_streams['vgg19'] = np.random.poisson(lambda_vgg19, self.time_duration) * 2
            self.input_streams['densenet201'] = np.random.poisson(lambda_densenet201, self.time_duration) * 2
            self.input_streams['mobilenet'] = np.random.poisson(lambda_mobilenet, self.time_duration) * 2
        else:
            self.input_streams = collections.defaultdict(list)
            self.input_streams['resnet50'] = self.tweet_input['resnet50']   #ascend在读取tweet输入流文件时已经加上了
            self.input_streams['vgg19'] = self.tweet_input['vgg19']
            self.input_streams['densenet201'] = self.tweet_input['densenet201']
            self.input_streams['mobilenet'] = self.tweet_input['mobilenet']

        self.input_streams_sum = collections.defaultdict(int)
        self.input_streams_sum['resnet50'] = sum(self.input_streams['resnet50'])
        self.input_streams_sum['vgg19'] = sum(self.input_streams['vgg19'])
        self.input_streams_sum['densenet201'] = sum(self.input_streams['densenet201'])
        self.input_streams_sum['mobilenet'] = sum(self.input_streams['mobilenet'])

    # 在ppo文件中使用，为了获取当前的输入流信息绘制到tensorboard
    def getInputStreamMessage(self):
        if not TWEET_INPUT:
            return "poisson", self.lambda_base
        else:
            return "tweet", self.input_streams
    
    # 在evaluation文件中使用，获取玩一次游戏的具体输入流信息
    def getOncTimeInputStreamMessage(self):
        return self.input_streams

    # 重置丢弃队列、模型load计数器状态、模型实际执行时间记录器状态
    def defineDiscardRequest(self):
        self.discard_request_nums = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.discard_request_nums['resnet50'] = [0 for _ in range(self.time_duration)]
        self.discard_request_nums['vgg19'] = [0 for _ in range(self.time_duration)]
        self.discard_request_nums['densenet201'] = [0 for _ in range(self.time_duration)]
        self.discard_request_nums['mobilenet'] = [0 for _ in range(self.time_duration)]

        self.load_model_times = collections.defaultdict(int)
        self.load_model_times['resnet50'] = 0
        self.load_model_times['vgg19'] = 0
        self.load_model_times['densenet201'] = 0
        self.load_model_times['mobilenet'] = 0

         # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
        self.exec_time = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.exec_time['resnet50'] = [0 for _ in range(self.time_duration)]
        self.exec_time['vgg19'] = [0 for _ in range(self.time_duration)]
        self.exec_time['densenet201'] = [0 for _ in range(self.time_duration)]
        self.exec_time['mobilenet'] = [0 for _ in range(self.time_duration)]

    # 初始化各个GPU上的模型load状态(一开始都没load)并定义load每个模型的cost
    def defineModelCacheState(self):
        self.cache_state = collections.defaultdict(list)
        self.cache_state['resnet50'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['vgg19'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['densenet201'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['mobilenet'] = [0 for _ in range(GPU_NUMS)]

        self.loading_cost = collections.defaultdict(int)
        self.loading_cost['resnet50'] = 2.0 * 5
        self.loading_cost['vgg19'] = 2.5 * 5
        self.loading_cost['densenet201'] = 2.5 * 5
        self.loading_cost['mobilenet'] = 2.0 * 5


    # 重置存储Throughput的字典信息，checkThroughput中将调用该函数对各个吞吐量赋值
    def defineThroughputState(self):
        self.throughput = collections.defaultdict(int)
        self.throughput['resnet50'] = 0
        self.throughput['vgg19'] = 0
        self.throughput['densenet201'] = 0
        self.throughput['mobilenet'] = 0

        self.throughput_list = collections.defaultdict(int)
        self.throughput_list['resnet50'] = []
        self.throughput_list['vgg19'] = []
        self.throughput_list['densenet201'] = []
        self.throughput_list['mobilenet'] = []


    # 定义记录每个模型最近在哪个GPU上执行的列表
    def defineRecentUsedLabel(self):
        self.recent_use_label = collections.defaultdict(list)
        self.recent_use_label['resnet50'] = []
        self.recent_use_label['vgg19'] = []
        self.recent_use_label['densenet201'] = []
        self.recent_use_label['mobilenet'] = []

    # 初始化evalution所需的各个数据字典
    def defineEvaluationDict(self):
        self.extra_message = collections.defaultdict(dict)

        self.throughput_eva_dict = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.extra_message['throughput'] = self.throughput_eva_dict                             # 每个模型每个时刻的吞吐量

        self.buffer_eva_dict = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.extra_message['buffer'] = self.buffer_eva_dict                                     # 每个模型每个时刻buffer内请求数量

        self.extra_message['discard'] = self.discard_request_nums                               # 每个模型每个时刻请求丢弃数量

        self.load_cost_eva_dict = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.extra_message['GPUloadcost'] = self.load_cost_eva_dict                             # 每个GPU每个时刻的loadmodel cost

        self.busy_time_eva_dict = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.extra_message['GPUbusytime'] = self.busy_time_eva_dict                             # 每个GPU每个时刻的时间占用长度

    # 展示buffer情况
    def showBufferDetails(self):
        for model_name in self.model_names:
            cur_buffer = self.buffers[model_name]
            print("---------- cur buffer details ----------")
            print("model name:", model_name)
            for request_group in cur_buffer:
                request_group.showDetails()

    # 展示cache情况
    def showCacheDetails(self):
        print("*** cur cache state ***")
        for model_name in self.model_names:
            cur_cache = self.cache_state[model_name]
            print("{}:[{},{}]".format(model_name, cur_cache[0], cur_cache[1]))
        
    # 执行该函数获得某时刻的state
    def getStateFunc(self, next_time):
        # state的定义是：（模型的请求到达数 + 对应的buffer内请求数 + 是否在GPU1缓存 + 是否在GPU2缓存）× 模型数量
        # 在基础架构下是4×4=16个数字
        state_vector = []
        for model_name in self.model_names:
            next_input = self.input_streams[model_name][next_time]
            cur_model_buffer = self.buffers[model_name]
            buffer_request_nums = 0
            for rq in cur_model_buffer:
                buffer_request_nums += rq.cur_requset_nums
            # 模型的请求到达数
            state_vector.append(next_input)
            # 对应的buffer内请求数
            state_vector.append(buffer_request_nums)
            # 是否在GPU1缓存
            cached_on_gpu1 = self.cache_state[model_name][0]
            state_vector.append(cached_on_gpu1)
            # 是否在GPU2缓存
            cached_on_gpu2 = self.cache_state[model_name][1]
            state_vector.append(cached_on_gpu2)
        return torch.Tensor(state_vector)
    
    # 执行该函数则在对应GPU上load对应模型，并返回cost
    def loadModelOnGPU(self, model_name, gpu_idx):
        # 如果对应GPU上已经加载了模型，则不load，返回cost=0
        if self.cache_state[model_name][gpu_idx] == 1:
            return 0
        # 否则load对应模型并返回cost
        else:
            self.cache_state[model_name][gpu_idx] = 1
            # print("successfully load {} on gpu {}, cost += {}".format(model_name, gpu_idx, self.loading_cost[model_name]))
            self.load_model_times[model_name] += 1
            return self.loading_cost[model_name]

    # 执行该函数则在特定GPU上unload特定模型
    def unloadModelFromGPU(self, model_name, gpu_idx):
        # unload在特定GPU上的特定模型，unload是瞬时的，没有cost
        self.cache_state[model_name][gpu_idx] = 0
        # print("unload {} from gpu {}...".format(model_name, gpu_idx))

    # 执行该函数检查对应模型(根据模型名)是否满足unload条件，如果满足条件，则检查应该unload哪个GPU上的缓存
    def unloadModelChecking(self, model_name):
        def judgeBothGPUloading(self):
            # if sum(self.cache_state[model_name]) == GPU_NUMS:
                # print("{} both load on GPU".format(model_name))
            return sum(self.cache_state[model_name]) == GPU_NUMS
        
        def judgeBufferEmpty(self):
            # if len(self.buffers[model_name]) == 0:
                # print("{} buffer empty".format(model_name))
            return len(self.buffers[model_name]) == 0
        
        def judgeRequest(self, next_time_length):
            cur_throughput = max(self.throughput_list[model_name])
            for t in range(self.time, self.time + next_time_length):
                if t < self.time_duration:
                    if cur_throughput < self.input_streams[model_name][t]:
                        return False
                else:
                    return False
            # print("{} request ok".format(model_name))
            return True

        def getRecentUsedGPUIndex(self):
            return self.recent_use_label[model_name][-1]

        # unload模型需要考虑多个因素：该模型是否在两个GPU上都加载了、该模型接下来一段时间内的请求到达率是否满足要求(小于当前时刻吞吐量)、该模型的buffer是否为空
        # 如果当前模型不是在两个GPU上都load过了，则不执行unload
        if not judgeBothGPUloading(self):
            return
        # 如果当前模型buffer不为空，则不执行unload
        if not judgeBufferEmpty(self):
            return
        # 如果当前模型请求到达率不满足要求(未来3个timescale的请求到达率小于当前时刻吞吐量)，则不执行unload
        if not judgeRequest(self, 10):
            return
        
        # 上述条件都满足，则找到该模型最近没有使用过的那个GPU编号
        recent_used_gpu_idx = getRecentUsedGPUIndex(self)
        # 只有在两个GPU时可以这么计算：获得最近没使用的那个GPU编号(如果最近使用的是0，则没使用的是(0+1)%2=1，如果最近使用的是1，则没使用的是(1+1)%2=0)
        unload_gpu_idx = (recent_used_gpu_idx + 1) % 2  
        # 执行卸载
        self.unloadModelFromGPU(model_name, unload_gpu_idx)

    # 执行该函数获取动作和索引之间的映射
    def getActionIndexMapping(self):
        single_model_config = []
        for m_idx in range(len(self.model_names)):
            for g_idx in range(len(self.GPUS) + 1):     # 3.14修改：增加在两个GPU上都放置模型的动作，因此g_idx循环时增加1，表示两个GPU都放置模型的动作
                for b_idx in range(len(candidate_batch)):
                    single_model_config.append((self.model_names[m_idx], g_idx, candidate_batch[b_idx]))
        tmp_res = list(combinations(single_model_config, MODEL_NUMS))

        cnt = 0
        self.action_config_map = dict()
        for full_config in tmp_res:
            # print(full_config)
            name_set = set()
            for config in full_config:
                name_set.add(config[0])
            if len(name_set) == MODEL_NUMS:
                self.action_config_map[cnt] = full_config
                cnt += 1
        # 这里的return的目的是将该map传给其他文件，其他文件中(比如测试baseline)也需要动作和具体配置的映射。
        # 在本环境文件中，用不到这个return，需要action_config_map的地方直接使用self.action_config_map获取即可
        return self.action_config_map


    def __init__(self, evaluation_flag = False, discard_cost_alpha = 0.1, load_cost_alpha = 0.8, input_ascend = 0):
        # ----------------- 函数定义部分 ---------------------
        def getTweetInput(self):
            self.tweet_input = collections.defaultdict(list)
            with open("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\drl\\lstm\\tweet_load_base240.csv", mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    model_name = row[0]
                    for i in range(1, len(row)):
                        self.tweet_input[model_name].append(float(row[i]) + self.input_ascend)

        # ---------------- 实际初始化（一次性定义部分） -------------------
        # 3.1修改action定义：action space: GPU_NUMS * len(candidate_batch)
        # 3.14修改action定义：action space: (GPU_NUMS+1) * len(candidate_batch) 增加在两个GPU上都放置模型的动作(因为维度问题，这两个动作batch一致)，则可选择的“GPU”为idx=1，idx=2，以及idx=1和2，因此GPU_NUMS+1
        self.action_length = int(math.pow((1+GPU_NUMS) * len(candidate_batch), MODEL_NUMS))
        self.action_space = spaces.Discrete(self.action_length)

        # 4模型时
        # observation space: （模型当前时刻请求数 + 模型当前时刻BUFFER队列长度 + 模型在GPU1是否缓存 + GPU2是否缓存 +...+ GPUN是否缓存）* 模型数量
        self.observation_length = int((2 + GPU_NUMS) * MODEL_NUMS)
        self.observation_space = spaces.MultiDiscrete([800, 800, 800, 800,
                                                       500, 500, 500, 500,
                                                       2, 2, 2, 2, 
                                                       2, 2, 2, 2], dtype=np.float32)

        # 6模型时
        # observation space: （模型当前时刻请求数 + 模型当前时刻BUFFER队列长度 + 模型在GPU1是否缓存 + GPU2是否缓存 +...+ GPUN是否缓存）* 模型数量
        # self.observation_length = int((2 + GPU_NUMS) * MODEL_NUMS)
        # self.observation_space = spaces.MultiDiscrete([800, 800, 800, 800, 800, 800,
        #                                                500, 500, 500, 500, 500, 500,
        #                                                2, 2, 2, 2, 2, 2,
        #                                                2, 2, 2, 2, 2, 2], dtype=np.float32)

        self.model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']                                              
        self.CART_dtr = joblib.load("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\\drl\\cart.pkl") # 读取
        self.request_retention_time_scale = 3           # 表示请求经过多少个time scale之后没被处理的话就丢弃了
        self.time_scale = 1                             # 代表1s
        self.time_duration = 240
        self.evaluation = evaluation_flag
        self.load_cost_alpha = load_cost_alpha          # 请求丢失时计算负reward的系数
        self.discard_cost_alpha = discard_cost_alpha    # 加载模型时，计算加载损失reward的系数
        self.input_ascend = input_ascend


        # ---------------- 实际初始化（游戏结束需要重置的部分） -------------------
        # 初始化GPU状态
        self.GPUS = [GPU(i) for i in range(GPU_NUMS)]
        # 表示执行过程的第几个time scale
        self.time = 0  
        # 重置buffer                                 
        self.defineBuffers()
        # 获取tweet输入
        if TWEET_INPUT:
            getTweetInput(self)
        # 初始化输入流
        self.defineInputStream()
        # 初始化模型缓存情况
        self.defineModelCacheState()
        # 初始化丢弃请求计数器
        self.defineDiscardRequest()
        # 初始化模型最近使用的GPU信息列表
        self.defineRecentUsedLabel()
        # 初始化evaluation所需的各个字典
        self.defineEvaluationDict()
        # 获取实际动作与动作索引的映射关系
        _= self.getActionIndexMapping()
        
        # print("gpucluste1a_p_4m: This is {}".format(os.path.abspath(os.path.dirname(__file__))))

    def step_igniter(self, action): 
        # 执行该函数输出所有GPU上的所有place信息
        def showAllPlaceDetails(self):
            for GPU in self.GPUS:
                GPU.showPlaces()

        # 执行该函数则检查当前所有模型buffer中是否有过期请求，如果有过期请求则删除它们并记录SLO
        def checkExpiredRequestInBuffer(self):
            discarded_request_nums = -10
            for model_name, buffer in self.buffers.items():
                new_buffer = []
                for group in buffer:
                    # 如果当前时间-请求到达时间>设置的保留时长，丢弃该group
                    if self.time - group.create_time >= self.request_retention_time_scale:
                        # 丢弃前对丢弃的请求进行记录：t=group.create_time时刻，model_name的cur_requset_nums个请求没有被处理，触发SLO

                        # ------- evaluation part -------   记录当前模型在请求创建时刻创建的请求中，最后会被discard的请求数量
                        self.discard_request_nums[group.model_name][group.create_time] += group.cur_requset_nums
                        # ------------ end --------------

                        discarded_request_nums += group.cur_requset_nums
                        continue
                    else:
                        new_buffer.append(group)
                self.buffers[model_name] = new_buffer
            return discarded_request_nums
        
        # 执行该函数则根据当前action在各个GPU上放置places
        def placeAccordingToActions(self, action):
            # if len(action) != MODEL_NUMS or len(self.model_names) != MODEL_NUMS:
            #     print("每时刻动作数量与模型种类数不符~")
            #     return
            config = action
            # config = [[[2, 'densenet201', 10, 57], [1, 'vgg19', 9, 42]], [[0, 'resnet50', 12, 24], [3, 'mobilenet', 19, 9]]]
            # config第一层表示两个GPU，第二层表示每个GPU上每个模型的名称、batch、资源量
            tmp_gpus = [[] for _ in range(len(self.GPUS))]
            for i in range(len(config)):
                cur_model_name = config[i][0]
                cur_model_gpu_idx = config[i][1]
                cur_model_batch = config[i][2]
                # 如果cur_model_gpu_idx=2说明动作要把该配置放在两个GPU上，则在两个GPU上都增加place
                if cur_model_gpu_idx == 2:
                    tmp_gpus[0].append((self.model_names[i], cur_model_batch))
                    tmp_gpus[1].append((self.model_names[i], cur_model_batch))
                else:
                    tmp_gpus[cur_model_gpu_idx].append((self.model_names[i], cur_model_batch))
            
            for gpu_idx, tmp_gpu in enumerate(tmp_gpus):
                n = len(tmp_gpu)
                tmp_gpu_r = []
                if n == 1:
                    tmp_gpu_r = [100]
                elif n == 2:
                    tmp_gpu_r = [50, 50]
                elif n == 3:
                    tmp_gpu_r = [30, 30, 40]
                elif n == 4:
                    tmp_gpu_r = [20, 20, 30, 30]
                elif n == 5:
                    tmp_gpu_r = [20, 20, 20, 20, 20]
                elif n == 6:
                    tmp_gpu_r = [10, 10, 10, 10, 10, 10]
                for idx, model_data in enumerate(tmp_gpu):
                    cur_model_name = model_data[0]
                    cur_model_batch = model_data[1]
                    cur_model_gpu_r = tmp_gpu_r[idx]
                    new_model_place = ModelPlace(cur_model_name, cur_model_batch, cur_model_gpu_r, gpu_idx)
                    self.GPUS[gpu_idx].addPlaces(new_model_place)

        # 执行该函数则检查当前action表示的动作是否可行（GPU资源量大于100则不可行），需要先执行placeAccordingToActions，放置后再检查
        def checkActionFeasibility(self):
            for idx, GPU in enumerate(self.GPUS):
                # print("GPU{}:".format(idx))
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
        def takeRequsetOutOfBuffer(model_name, take_nums):
            buffer = self.buffers[model_name]
            i = 0
            actual_inf = 0

            buffered_request_sum = 0
            # 首先遍历计算当前buffer中该模型有多少缓存的请求数
            for idx, group in enumerate(buffer):
                buffered_request_sum += group.cur_requset_nums
            # 如果缓存的请求数小于take_nums(当前可处理的)，说明可将缓存中的全部处理，则记录实际推理数量actual_inf为缓存内的请求数，然后直接清空buffer
            if buffered_request_sum <= take_nums:
                actual_inf = buffered_request_sum
                self.buffers[model_name] = []
            # 否则逐个查看requestGroup，将可以处理完的group删除掉
            else:
                for idx, group in enumerate(buffer):
                    if take_nums > 0:
                        if group.cur_requset_nums > take_nums:
                            group.cur_requset_nums -= take_nums
                            actual_inf += take_nums
                            take_nums = 0
                        else:
                            take_nums -= group.cur_requset_nums
                            actual_inf += group.cur_requset_nums
                        i = idx
                self.buffers[model_name] = buffer[i:]

            return actual_inf

        # 执行该函数则检查各个模型的吞吐量是否达标，是否需要buffer增减
        def checkThroughput(self):
            # 定义一个最后要最小化的变量t，t表示当前时刻所有GPU上所有place的执行时间之和
            t = 0
            self.defineThroughputState()
            # 建立一个存储当前模型都在哪些GPU上有place的数据结构
            where_place = collections.defaultdict(list)

            for idx, GPU in enumerate(self.GPUS):
                for place in GPU.places:
                    model_name = place.model_name
                    # 记录吞吐量
                    self.throughput[model_name] += place.throughput
                    self.throughput_list[model_name].append(place.throughput)
                    # 记录当前模型在GPU idx上有一个吞吐量为place.throughput的place
                    where_place[model_name].append([idx, place.throughput])

                    # 记录当前模型最近在编号为idx的GPU上执行了
                    self.recent_use_label[model_name].append(idx)
                    # t增加load模型带来的cost，如果当前GPU上已有model_name模型，则self.loadModelOnGPU返回0
                    cur_model_load_cost = self.loadModelOnGPU(model_name, idx)
                    t += cur_model_load_cost

                    # ------- evaluation part -------   记录当前GPU在当前时刻t的load cost
                    self.load_cost_eva_dict[idx][self.time] += cur_model_load_cost
                    # ------------ end --------------
            
            # ------- evaluation part -------   记录每个模型在当前时刻t的吞吐量
            for model_name in self.model_names:
                self.throughput_eva_dict[model_name][self.time] = self.throughput[model_name]
            # ------------ end --------------

            for model_name in self.model_names:
                model_request_nums = self.input_streams[model_name][self.time]
                model_throughput = self.throughput[model_name]
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
                    # 3.14新增模型在两个GPU都可能放置，因此这里增加判断：

                    if len(where_place[model_name]) == 2:
                        # 如果where_place[model_name]的长度为2，则说明两个GPU上都放置了该模型的place，此时还处理不完，则应该记录该模型执行时间为2个timescale
                        # 然而，reward t不应该改变，仍为一个timescale，因为定义两个GPU都可放置模型的动作就是为了找到更合适的调度方式，现在执行这种调度方式，回报应该是该调度方式下实际物理时间过了多久
                        # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
                        self.exec_time[model_name][self.time] += 2 * self.time_scale
                    else:
                        self.exec_time[model_name][self.time] += self.time_scale

                    # 实际的reward t 仍为一倍的timescale
                    t += self.time_scale

                    # ------- evaluation part -------   记录每个GPU的最大繁忙时间
                    # 首先获取当前model_name在哪个GPU上有place
                    for place in where_place[model_name]:
                        # 获取每个place的gpu索引
                        gpu_idx = place[0]
                        # 记录当前gpu的执行时间为1个timescale，并与之前记录的值对比，取更大的值覆盖
                        self.busy_time_eva_dict[gpu_idx][self.time] = max(self.busy_time_eva_dict[gpu_idx][self.time], self.time_scale)
                    # ------------ end --------------

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
                        buffer_inf = takeRequsetOutOfBuffer(model_name, remaining_throughput_nums)
                        # 根据实际推理数量计算推理时长
                        # （时间间隔 / 吞吐量 = 推理一次的时间） × （本次时间间隔内的推理数量+buffer中取出的请求推理数量）
                    
                    # 3.14新增模型在两个GPU都可能放置，因此这里增加判断：
                    if len(where_place[model_name]) == 2:
                        # 如果where_place[model_name]的长度为2，则说明两个GPU上都放置了该模型的place，此时应该记录模型执行时间为两个GPU上执行时间的和，这里直接用2倍的reward近似该值
                        # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
                        self.exec_time[model_name][self.time] += (2 * self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    else:
                        self.exec_time[model_name][self.time] += (self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    
                    # 实际的reward t 仍为一倍的reward=(self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    t += (self.time_scale / model_throughput) * (model_request_nums + buffer_inf)

                    # ------- evaluation part -------   记录每个GPU的最大繁忙时间
                    # 首先获取当前model_name在哪个GPU上有place
                    for place in where_place[model_name]:
                        # 获取每个place的gpu索引
                        gpu_idx = place[0]
                        # 记录当前gpu的执行时间为1个timescale，并与之前记录的值对比，取更大的值覆盖
                        self.busy_time_eva_dict[gpu_idx][self.time] = max(self.busy_time_eva_dict[gpu_idx][self.time], self.time_scale)
                    # ------------ end --------------

            # ------- evaluation part -------   记录每个模型在当前时刻t的buffer内请求数
            for model_name in self.model_names:
                request_nums = 0
                buffer = self.buffers[model_name]
                for rq in buffer:
                    request_nums += rq.cur_requset_nums
                self.buffer_eva_dict[model_name][self.time] = request_nums
            # ------------ end --------------

            return t
        
        # 执行该函数则检查所有模型是否需要卸载
        def checkUnloading(self):
            # self.showCacheDetails()
            # print("--- 卸载后 --")
            for model_name in self.model_names:
                self.unloadModelChecking(model_name)
            # self.showCacheDetails()

        # 执行该函数则清空所有GPU上的place信息，这个函数应该在每次step结束之前调用且必须调用
        def clearGPUPlaces(self):
            for GPU in self.GPUS:
                GPU.deletePlaces()
        # ---------------- 函数执行部分 -----------------
        done = False
        reward = 0
        reward -= 0.1 * checkExpiredRequestInBuffer(self)
        placeAccordingToActions(self, action)
        getRealThroughput(self)
        # showAllPlaceDetails(self)
        reward -= checkThroughput(self)
        # self.showBufferDetails()

        checkUnloading(self)
        clearGPUPlaces(self)
        
        next_state = np.array([])
        if self.time + 1 < self.time_duration:
            next_state = np.array(self.getStateFunc(self.time + 1))
        else:
            done = True
        self.time += 1
        # reward /= self.time
        # reward += self.time * 100

        resnet50_dis_nums = sum(self.discard_request_nums['resnet50'])
        resnet50_dis_percent = resnet50_dis_nums / self.input_streams_sum['resnet50']
        resnet50_load_times = self.load_model_times['resnet50']
        resnet50_exec_time = sum(self.exec_time['resnet50'])

        vgg19_dis_nums = sum(self.discard_request_nums['vgg19'])
        vgg19_dis_percent = vgg19_dis_nums / self.input_streams_sum['vgg19']
        vgg19_load_times = self.load_model_times['vgg19']
        vgg19_exec_time = sum(self.exec_time['vgg19'])

        densenet201_dis_nums = sum(self.discard_request_nums['densenet201'])
        densenet201_dis_percent = densenet201_dis_nums / self.input_streams_sum['densenet201']
        densenet201_load_times = self.load_model_times['densenet201']
        densenet201_exec_time = sum(self.exec_time['densenet201'])

        mobilenet_dis_nums = sum(self.discard_request_nums['mobilenet'])
        mobilenet_dis_percent = mobilenet_dis_nums / self.input_streams_sum['mobilenet']
        mobilenet_load_times = self.load_model_times['mobilenet']
        mobilenet_exec_time = sum(self.exec_time['mobilenet'])

        if self.evaluation and done:
            dis_per_load_times = {
                # dis percent和loadtimes 在每一轮迭代中都会计算，但此处返回的是最后一轮迭代得到的这两个值
                # 按照定义，dis_percent此时使用的是所有时刻t的dis nums之和除以总对应的模型请求到达数量，代表了整个时序流中的dis情况
                # 而loadtimes是每一轮迭代累加，在最后一轮将得到整个时序流中每个模型总的loadtimes数量，因此都符合evaluate标准
                'resnet50_dis_percent':resnet50_dis_percent,
                'vgg19_dis_percent':vgg19_dis_percent,
                'densenet201_dis_percent':densenet201_dis_percent,
                'mobilenet_dis_percent':mobilenet_dis_percent,

                'resnet50_load_times':resnet50_load_times,
                'vgg19_load_times':vgg19_load_times,
                'densenet201_load_times':densenet201_load_times,
                'mobilenet_load_times':mobilenet_load_times
            }
            return  next_state, reward, done, (self.extra_message, dis_per_load_times)
        
        return next_state, reward, done, {'time': self.time - 1, 
                                          "resnet50_dis_nums": resnet50_dis_nums, "resnet50_dis_percent": resnet50_dis_percent,
                                          "resnet50_load_times": resnet50_load_times, "resnet50_exec_time": resnet50_exec_time,
                                          "vgg19_dis_nums": vgg19_dis_nums, "vgg19_dis_percent": vgg19_dis_percent,
                                          "vgg19_load_times": vgg19_load_times, "vgg19_exec_time": vgg19_exec_time,
                                          "densenet201_dis_nums": densenet201_dis_nums, "densenet201_dis_percent": densenet201_dis_percent, 
                                          "densenet201_load_times": densenet201_load_times, "densenet201_exec_time": densenet201_exec_time,
                                          "mobilenet_dis_nums": mobilenet_dis_nums, "mobilenet_dis_percent": mobilenet_dis_percent, 
                                          "mobilenet_load_times": mobilenet_load_times, "mobilenet_exec_time": mobilenet_exec_time}
    
    def step(self, action): 
        # 执行该函数输出所有GPU上的所有place信息
        def showAllPlaceDetails(self):
            for GPU in self.GPUS:
                GPU.showPlaces()

        # 执行该函数则检查当前所有模型buffer中是否有过期请求，如果有过期请求则删除它们并记录SLO
        def checkExpiredRequestInBuffer(self):
            discarded_request_nums = -10
            for model_name, buffer in self.buffers.items():
                new_buffer = []
                for group in buffer:
                    # 如果当前时间-请求到达时间>设置的保留时长，丢弃该group
                    if self.time - group.create_time >= self.request_retention_time_scale:
                        # 丢弃前对丢弃的请求进行记录：t=group.create_time时刻，model_name的cur_requset_nums个请求没有被处理，触发SLO

                        # ------- evaluation part -------   记录当前模型在请求创建时刻创建的请求中，最后会被discard的请求数量
                        self.discard_request_nums[group.model_name][group.create_time] += group.cur_requset_nums
                        # ------------ end --------------

                        discarded_request_nums += group.cur_requset_nums
                        continue
                    else:
                        new_buffer.append(group)
                self.buffers[model_name] = new_buffer
            return discarded_request_nums
        
        # 执行该函数则根据当前action在各个GPU上放置places
        def placeAccordingToActions(self, action):
            # if len(action) != MODEL_NUMS or len(self.model_names) != MODEL_NUMS:
            #     print("每时刻动作数量与模型种类数不符~")
            #     return
            config = self.action_config_map[action]
            # config: (['resnet50', 1, 8], ['vgg19', 1, 8], ['densenet201', 0, 8], ['mobilenet', 0, 16])
            # print(config)
            tmp_gpus = [[] for _ in range(len(self.GPUS))]
            for i in range(len(config)):
                cur_model_name = config[i][0]
                cur_model_gpu_idx = config[i][1]
                cur_model_batch = config[i][2]
                # 如果cur_model_gpu_idx=2说明动作要把该配置放在两个GPU上，则在两个GPU上都增加place
                if cur_model_gpu_idx == 2:
                    tmp_gpus[0].append((self.model_names[i], cur_model_batch))
                    tmp_gpus[1].append((self.model_names[i], cur_model_batch))
                else:
                    tmp_gpus[cur_model_gpu_idx].append((self.model_names[i], cur_model_batch))
            
            for gpu_idx, tmp_gpu in enumerate(tmp_gpus):
                n = len(tmp_gpu)
                tmp_gpu_r = []
                if n == 1:
                    tmp_gpu_r = [100]
                elif n == 2:
                    tmp_gpu_r = [50, 50]
                elif n == 3:
                    tmp_gpu_r = [30, 30, 40]
                elif n == 4:
                    tmp_gpu_r = [20, 20, 30, 30]
                elif n == 5:
                    tmp_gpu_r = [20, 20, 20, 20, 20]
                elif n == 6:
                    tmp_gpu_r = [10, 10, 10, 10, 10, 10]
                for idx, model_data in enumerate(tmp_gpu):
                    cur_model_name = model_data[0]
                    cur_model_batch = model_data[1]
                    cur_model_gpu_r = tmp_gpu_r[idx]
                    new_model_place = ModelPlace(cur_model_name, cur_model_batch, cur_model_gpu_r, gpu_idx)
                    self.GPUS[gpu_idx].addPlaces(new_model_place)

        # 执行该函数则检查当前action表示的动作是否可行（GPU资源量大于100则不可行），需要先执行placeAccordingToActions，放置后再检查
        def checkActionFeasibility(self):
            for idx, GPU in enumerate(self.GPUS):
                # print("GPU{}:".format(idx))
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
        def takeRequsetOutOfBuffer(model_name, take_nums):
            buffer = self.buffers[model_name]
            i = 0
            actual_inf = 0

            buffered_request_sum = 0
            # 首先遍历计算当前buffer中该模型有多少缓存的请求数
            for idx, group in enumerate(buffer):
                buffered_request_sum += group.cur_requset_nums
            # 如果缓存的请求数小于take_nums(当前可处理的)，说明可将缓存中的全部处理，则记录实际推理数量actual_inf为缓存内的请求数，然后直接清空buffer
            if buffered_request_sum <= take_nums:
                actual_inf = buffered_request_sum
                self.buffers[model_name] = []
            # 否则逐个查看requestGroup，将可以处理完的group删除掉
            else:
                for idx, group in enumerate(buffer):
                    if take_nums > 0:
                        if group.cur_requset_nums > take_nums:
                            group.cur_requset_nums -= take_nums
                            actual_inf += take_nums
                            take_nums = 0
                        else:
                            take_nums -= group.cur_requset_nums
                            actual_inf += group.cur_requset_nums
                        i = idx
                self.buffers[model_name] = buffer[i:]

            return actual_inf

        # 执行该函数则检查各个模型的吞吐量是否达标，是否需要buffer增减
        def checkThroughput(self):
            # 定义一个最后要最小化的变量t，t表示当前时刻所有GPU上所有place的执行时间之和
            t = 0
            self.defineThroughputState()
            # 建立一个存储当前模型都在哪些GPU上有place的数据结构
            where_place = collections.defaultdict(list)

            for idx, GPU in enumerate(self.GPUS):
                for place in GPU.places:
                    model_name = place.model_name
                    # 记录吞吐量
                    self.throughput[model_name] += place.throughput
                    self.throughput_list[model_name].append(place.throughput)
                    # 记录当前模型在GPU idx上有一个吞吐量为place.throughput的place
                    where_place[model_name].append([idx, place.throughput])

                    # 记录当前模型最近在编号为idx的GPU上执行了
                    self.recent_use_label[model_name].append(idx)
                    # t增加load模型带来的cost，如果当前GPU上已有model_name模型，则self.loadModelOnGPU返回0
                    cur_model_load_cost = self.loadModelOnGPU(model_name, idx)
                    t += cur_model_load_cost

                    # ------- evaluation part -------   记录当前GPU在当前时刻t的load cost
                    self.load_cost_eva_dict[idx][self.time] += cur_model_load_cost
                    # ------------ end --------------
            
            # ------- evaluation part -------   记录每个模型在当前时刻t的吞吐量
            for model_name in self.model_names:
                self.throughput_eva_dict[model_name][self.time] = self.throughput[model_name]
            # ------------ end --------------

            for model_name in self.model_names:
                model_request_nums = self.input_streams[model_name][self.time]
                model_throughput = self.throughput[model_name]
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
                    # 3.14新增模型在两个GPU都可能放置，因此这里增加判断：

                    if len(where_place[model_name]) == 2:
                        # 如果where_place[model_name]的长度为2，则说明两个GPU上都放置了该模型的place，此时还处理不完，则应该记录该模型执行时间为2个timescale
                        # 然而，reward t不应该改变，仍为一个timescale，因为定义两个GPU都可放置模型的动作就是为了找到更合适的调度方式，现在执行这种调度方式，回报应该是该调度方式下实际物理时间过了多久
                        # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
                        self.exec_time[model_name][self.time] += 2 * self.time_scale
                    else:
                        self.exec_time[model_name][self.time] += self.time_scale

                    # 实际的reward t 仍为一倍的timescale
                    t += self.time_scale

                    # ------- evaluation part -------   记录每个GPU的最大繁忙时间
                    # 首先获取当前model_name在哪个GPU上有place
                    for place in where_place[model_name]:
                        # 获取每个place的gpu索引
                        gpu_idx = place[0]
                        # 记录当前gpu的执行时间为1个timescale，并与之前记录的值对比，取更大的值覆盖
                        self.busy_time_eva_dict[gpu_idx][self.time] = max(self.busy_time_eva_dict[gpu_idx][self.time], self.time_scale)
                    # ------------ end --------------

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
                        buffer_inf = takeRequsetOutOfBuffer(model_name, remaining_throughput_nums)
                        # 根据实际推理数量计算推理时长
                        # （时间间隔 / 吞吐量 = 推理一次的时间） × （本次时间间隔内的推理数量+buffer中取出的请求推理数量）
                    
                    # 3.14新增模型在两个GPU都可能放置，因此这里增加判断：
                    if len(where_place[model_name]) == 2:
                        # 如果where_place[model_name]的长度为2，则说明两个GPU上都放置了该模型的place，此时应该记录模型执行时间为两个GPU上执行时间的和，这里直接用2倍的reward近似该值
                        # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
                        self.exec_time[model_name][self.time] += (2 * self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    else:
                        self.exec_time[model_name][self.time] += (self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    
                    # 实际的reward t 仍为一倍的reward=(self.time_scale / model_throughput) * (model_request_nums + buffer_inf)
                    t += (self.time_scale / model_throughput) * (model_request_nums + buffer_inf)

                    # ------- evaluation part -------   记录每个GPU的最大繁忙时间
                    # 首先获取当前model_name在哪个GPU上有place
                    for place in where_place[model_name]:
                        # 获取每个place的gpu索引
                        gpu_idx = place[0]
                        # 记录当前gpu的执行时间为1个timescale，并与之前记录的值对比，取更大的值覆盖
                        self.busy_time_eva_dict[gpu_idx][self.time] = max(self.busy_time_eva_dict[gpu_idx][self.time], self.time_scale)
                    # ------------ end --------------

            # ------- evaluation part -------   记录每个模型在当前时刻t的buffer内请求数
            for model_name in self.model_names:
                request_nums = 0
                buffer = self.buffers[model_name]
                for rq in buffer:
                    request_nums += rq.cur_requset_nums
                self.buffer_eva_dict[model_name][self.time] = request_nums
            # ------------ end --------------

            return t
        
        # 执行该函数则检查所有模型是否需要卸载
        def checkUnloading(self):
            # self.showCacheDetails()
            # print("--- 卸载后 --")
            for model_name in self.model_names:
                self.unloadModelChecking(model_name)
            # self.showCacheDetails()

        # 执行该函数则清空所有GPU上的place信息，这个函数应该在每次step结束之前调用且必须调用
        def clearGPUPlaces(self):
            for GPU in self.GPUS:
                GPU.deletePlaces()
        # ---------------- 函数执行部分 -----------------
        done = False
        reward = 0
        reward -= 0.1 * checkExpiredRequestInBuffer(self)
        placeAccordingToActions(self, action)
        getRealThroughput(self)
        # showAllPlaceDetails(self)
        reward -= checkThroughput(self)
        # self.showBufferDetails()

        checkUnloading(self)
        clearGPUPlaces(self)
        
        next_state = np.array([])
        if self.time + 1 < self.time_duration:
            next_state = np.array(self.getStateFunc(self.time + 1))
        else:
            done = True
        self.time += 1
        # reward /= self.time
        # reward += self.time * 100

        resnet50_dis_nums = sum(self.discard_request_nums['resnet50'])
        resnet50_dis_percent = resnet50_dis_nums / self.input_streams_sum['resnet50']
        resnet50_load_times = self.load_model_times['resnet50']
        resnet50_exec_time = sum(self.exec_time['resnet50'])

        vgg19_dis_nums = sum(self.discard_request_nums['vgg19'])
        vgg19_dis_percent = vgg19_dis_nums / self.input_streams_sum['vgg19']
        vgg19_load_times = self.load_model_times['vgg19']
        vgg19_exec_time = sum(self.exec_time['vgg19'])

        densenet201_dis_nums = sum(self.discard_request_nums['densenet201'])
        densenet201_dis_percent = densenet201_dis_nums / self.input_streams_sum['densenet201']
        densenet201_load_times = self.load_model_times['densenet201']
        densenet201_exec_time = sum(self.exec_time['densenet201'])

        mobilenet_dis_nums = sum(self.discard_request_nums['mobilenet'])
        mobilenet_dis_percent = mobilenet_dis_nums / self.input_streams_sum['mobilenet']
        mobilenet_load_times = self.load_model_times['mobilenet']
        mobilenet_exec_time = sum(self.exec_time['mobilenet'])

        if self.evaluation and done:
            dis_per_load_times = {
                # dis percent和loadtimes 在每一轮迭代中都会计算，但此处返回的是最后一轮迭代得到的这两个值
                # 按照定义，dis_percent此时使用的是所有时刻t的dis nums之和除以总对应的模型请求到达数量，代表了整个时序流中的dis情况
                # 而loadtimes是每一轮迭代累加，在最后一轮将得到整个时序流中每个模型总的loadtimes数量，因此都符合evaluate标准
                'resnet50_dis_percent':resnet50_dis_percent,
                'vgg19_dis_percent':vgg19_dis_percent,
                'densenet201_dis_percent':densenet201_dis_percent,
                'mobilenet_dis_percent':mobilenet_dis_percent,

                'resnet50_load_times':resnet50_load_times,
                'vgg19_load_times':vgg19_load_times,
                'densenet201_load_times':densenet201_load_times,
                'mobilenet_load_times':mobilenet_load_times
            }
            return  next_state, reward, done, (self.extra_message, dis_per_load_times)
        
        return next_state, reward, done, {'time': self.time - 1, 
                                          "resnet50_dis_nums": resnet50_dis_nums, "resnet50_dis_percent": resnet50_dis_percent,
                                          "resnet50_load_times": resnet50_load_times, "resnet50_exec_time": resnet50_exec_time,
                                          "vgg19_dis_nums": vgg19_dis_nums, "vgg19_dis_percent": vgg19_dis_percent,
                                          "vgg19_load_times": vgg19_load_times, "vgg19_exec_time": vgg19_exec_time,
                                          "densenet201_dis_nums": densenet201_dis_nums, "densenet201_dis_percent": densenet201_dis_percent, 
                                          "densenet201_load_times": densenet201_load_times, "densenet201_exec_time": densenet201_exec_time,
                                          "mobilenet_dis_nums": mobilenet_dis_nums, "mobilenet_dis_percent": mobilenet_dis_percent, 
                                          "mobilenet_load_times": mobilenet_load_times, "mobilenet_exec_time": mobilenet_exec_time}

        # print("step: action:", action)
        # reward = 1
        # self.cnt += 1
        # done = False
        # if self.cnt >= 10:
        #     done = True
        # next_state = torch.rand(16)
        # return np.array(next_state, dtype=np.float32), reward, done, {}

    def reset(self):
        # 初始化部分需要对__init__函数中定义的诸多变量进行重置
        self.defineBuffers()
        self.defineInputStream()
        self.defineModelCacheState()
        self.defineDiscardRequest()
        self.defineRecentUsedLabel()
        self.defineEvaluationDict()
        self.GPUS = [GPU(i) for i in range(GPU_NUMS)]
        self.time = 0    

        # 获取初始状态
        origin_state = self.getStateFunc(0)
        return np.array(origin_state, dtype=np.float32)

# if __name__ == "__main__":
#     env = GPUEnv1a_p_4m_2g()
#     action = 400
#     env.step(action)
#     print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step1 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
#     action = 1200
#     env.step(action)
#     print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step2 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
#     action = 5321
#     env.step(action)
#     print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step3 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
#     action = 1100
#     env.step(action)
#     print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step4 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
#     action = 3201
#     env.step(action)
#     print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step5 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
