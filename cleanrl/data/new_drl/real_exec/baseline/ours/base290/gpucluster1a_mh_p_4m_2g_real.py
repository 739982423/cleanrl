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

# mobilenet_load_time 19.48134422302246
# vgg19_load_time 119.90809631347656
# densenet201_load_time 92.19647979736328
# resnet50_load_time 40.613887786865234


TWEET_INPUT = True
GPU_TYPE = "NVIDIA RTX2080TI"
GPU_PCIE_SPEED = 15754 * 1024   #KB/s
GPU_NUMS = 2

MODEL_NUMS = 4
candidate_batch = [4,8,16]
candidate_gpu_resource = [i * 10 for i in range(1, 11)]

global_kernel_data = collections.defaultdict(list)


# 读入的数据行每列的含义：Model,Batch,GPU Resource,L1RL2,L1WL2,L2RD,L2WD,L2Hit,L2Miss,Latency（与summary文件一致）
with open("/home/hpj/project/cleanrl/cleanrl/data/summary3.csv", mode="r", encoding="utf-8") as f:
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

class GPUEnv1a_mh_p_4m_2g_real(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    # 重置buffer状态
    def defineBuffers(self):
        self.buffers = collections.defaultdict(int)
        self.buffers['resnet50'] = 0
        self.buffers['vgg19'] = 0
        self.buffers['densenet201'] = 0
        self.buffers['mobilenet'] = 0
    
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
    
    # 重置丢弃队列、模型load计数器状态、模型unload计数器状态、模型实际执行时间记录器状态
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

        self.unload_model_times = collections.defaultdict(int)
        self.unload_model_times['resnet50'] = 0
        self.unload_model_times['vgg19'] = 0
        self.unload_model_times['densenet201'] = 0
        self.unload_model_times['mobilenet'] = 0

         # 记录每个模型的实际执行所用时长(返回给ppo文件画图用)
        self.exec_time = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.exec_time['resnet50'] = [0 for _ in range(self.time_duration)]
        self.exec_time['vgg19'] = [0 for _ in range(self.time_duration)]
        self.exec_time['densenet201'] = [0 for _ in range(self.time_duration)]
        self.exec_time['mobilenet'] = [0 for _ in range(self.time_duration)]

        # 记录每个模型的每个时刻可达到的吞吐量(返回给ppo文件画图用)
        self.model_throughput = collections.defaultdict(lambda:[0 for _ in range(self.time_duration)])
        self.model_throughput['resnet50'] = [0 for _ in range(self.time_duration)]
        self.model_throughput['vgg19'] = [0 for _ in range(self.time_duration)]
        self.model_throughput['densenet201'] = [0 for _ in range(self.time_duration)]
        self.model_throughput['mobilenet'] = [0 for _ in range(self.time_duration)]


    # 初始化各个GPU上的模型load状态(一开始都没load)并定义load每个模型的cost
    def defineModelCacheState(self):
        self.cache_state = collections.defaultdict(list)
        self.cache_state['resnet50'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['vgg19'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['densenet201'] = [0 for _ in range(GPU_NUMS)]
        self.cache_state['mobilenet'] = [0 for _ in range(GPU_NUMS)]

        # 3.24修改，此处的load_time为真实加载模型产生的时间开销
        self.real_loading_time = collections.defaultdict(int)
        self.real_loading_time['resnet50'] = 0.05
        self.real_loading_time['vgg19'] = 0.13
        self.real_loading_time['densenet201'] = 0.1
        self.real_loading_time['mobilenet'] = 0.025

        # 定义记录每个GPU因为缓存模型而使用的内存list
        self.gpu_cached_memory = [0 for _ in range(GPU_NUMS)]
        self.model_memory = collections.defaultdict(int)
        self.model_memory['resnet50'] = 100
        self.model_memory['vgg19'] = 561
        self.model_memory['densenet201'] = 79
        self.model_memory['mobilenet'] = 13
        self.max_memory_size = self.model_memory['resnet50'] + self.model_memory['vgg19'] + self.model_memory['densenet201'] + self.model_memory['mobilenet']

        # 定义记录每一时刻每个GPU上（最大模型缓存空间-当前模型缓存空间）的差，表示模型卸载腾出多少有效内存空间
        self.gpu_available_cache_memory_trace = [[0 for _ in range(2)] for _ in range(self.time_duration)]

    # 重置存储Throughput的字典信息，checkThroughput中将调用该函数对各个吞吐量赋值
    def defineThroughputState(self):
        self.throughput = collections.defaultdict(list)
        self.throughput['resnet50'] = [0 for _ in range(GPU_NUMS)]
        self.throughput['vgg19'] = [0 for _ in range(GPU_NUMS)]
        self.throughput['densenet201'] = [0 for _ in range(GPU_NUMS)]
        self.throughput['mobilenet'] = [0 for _ in range(GPU_NUMS)]

    # 在ppo文件中使用，为了获取当前的输入流信息绘制到tensorboard
    def getInputStreamMessage(self):
        if not TWEET_INPUT:
            return "poisson", self.lambda_base
        else:
            return "tweet", self.input_streams
    
    # 在evaluation文件中使用，获取玩一次游戏的具体输入流信息
    def getOncTimeInputStreamMessage(self):
        return self.input_streams

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
            print("{}, {}", model_name, cur_buffer)

    # 展示cache情况
    def showCacheDetails(self):
        print("*** cur cache state ***")
        for model_name in self.model_names:
            cur_cache = self.cache_state[model_name]
            print("{}:[{},{}]".format(model_name, cur_cache[0], cur_cache[1]))
        
    # 执行该函数获得某时刻的state
    def getStateFunc(self, next_time):
        # state的定义是：（模型的请求到达数 + 对应的buffer内请求数 + 是否在GPU1缓存 + 是否在GPU2缓存）× 模型数量 + GPU1缓存模型用的内存 + GPU2缓存模型用的内存
        # 在基础架构下是4×4+2=18个数字
        state_vector = []
        for model_name in self.model_names:
            next_input = self.input_streams[model_name][next_time]
            cur_model_buffer_length = self.buffers[model_name]
            # 模型的请求到达数
            state_vector.append(next_input)
            # 对应的buffer内请求数
            state_vector.append(cur_model_buffer_length)
            # 是否在GPU1缓存
            cached_on_gpu1 = self.cache_state[model_name][0]
            state_vector.append(cached_on_gpu1)
            # 是否在GPU2缓存
            cached_on_gpu2 = self.cache_state[model_name][1]
            state_vector.append(cached_on_gpu2)
            # GPU1和2缓存模型用的内存
            GPU1_cached_memory = self.gpu_cached_memory[0]
            GPU2_cached_memory = self.gpu_cached_memory[1]
        state_vector.append(GPU1_cached_memory)
        state_vector.append(GPU2_cached_memory)

        # 添加3个时刻历史输入流信息
        for i in range(1, 4):
            for model_name in self.model_names:
                if next_time - i >= 0:
                    state_vector.append(self.input_streams[model_name][next_time - i])
                else:
                    state_vector.append(0)
        # print(state_vector)
        # 添加next_time之后3个未来时刻输入流信息
        for i in range(1, 4):
            for model_name in self.model_names:
                if next_time + i < self.time_duration:
                    state_vector.append(self.input_streams[model_name][next_time + i])
                else:
                    state_vector.append(self.input_streams[model_name][next_time])

        return torch.Tensor(state_vector)

    # 执行该函数获取动作和索引之间的映射
    def getActionIndexMapping(self):
        single_model_config = []
        for m_idx in range(len(self.model_names)):
            for g_idx in range(len(self.GPUS) + 1):     # 3.14修改：增加在两个GPU上都放置模型的动作，因此g_idx循环时增加1，表示两个GPU都放置模型的动作
                for b_idx in range(len(candidate_batch)):
                    single_model_config.append((self.model_names[m_idx], g_idx, candidate_batch[b_idx]))
        tmp_res = list(combinations(single_model_config, MODEL_NUMS))

        cnt = 0
        self.exec_action_config_map = dict()
        for full_config in tmp_res:
            # print(full_config)
            name_set = set()
            for config in full_config:
                name_set.add(config[0])
            if len(name_set) == MODEL_NUMS:
                self.exec_action_config_map[cnt] = full_config
                cnt += 1
        
        # cache动作的映射字典，将cache动作的一个int数字映射为长度为8的0和1组成的list，代表加载和卸载后的2个GPU上4个模型的状态
        self.cache_action_config_map = dict()
        for i in range(self.cache_action_length):
            binary = list(bin(i)[2:])
            while len(binary) < GPU_NUMS * MODEL_NUMS:
                binary.insert(0,0)
            for idx, str in enumerate(binary):
                binary[idx] = int(str)
            self.cache_action_config_map[i] = binary

        # print(self.cache_action_config_map)
        # 这里的return的目的是将该map传给其他文件，其他文件中(比如测试baseline)也需要动作和具体配置的映射。
        # 在本环境文件中，用不到这个return，需要action_config_map的地方直接使用self.action_config_map获取即可
        return self.exec_action_config_map


    def __init__(self, evaluation_flag = False, discard_cost_alpha = 0.1, memory_reward_alpha = 0.005, input_ascend = 0):
        # ----------------- 函数定义部分 ---------------------
        def getTweetInput(self):
            self.tweet_input = collections.defaultdict(list)
            with open("/home/hpj/project/cleanrl/cleanrl/data/drl/lstm/tweet_load_base240.csv", mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    model_name = row[0]
                    for i in range(1, len(row)):
                        self.tweet_input[model_name].append(float(row[i]) + self.input_ascend)
        
        # 测试时使用的函数，获取真实动作下的实际吞吐量
        def evaGetRealResult():
            result_dir = "/home/hpj/project/cleanrl/cleanrl/data/new_drl/real_gpu_test/baseline/ours/base290/get_res/res/summary_data/"
            self.evaRealThroughput = collections.defaultdict(list)
            for model_name in self.model_names:
                with open(result_dir + model_name + ".csv", mode = "r", encoding = "utf-8-sig") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        gpu0_real_throughput = float(row[0])
                        gpu1_real_throughput = float(row[2])
                        self.evaGetRealResult[model_name].append([gpu0_real_throughput, gpu1_real_throughput])

        # ---------------- 实际初始化（一次性定义部分） -------------------
        # 3.1修改action定义：action space: GPU_NUMS * len(candidate_batch)
        # 3.14修改action定义：action space: (GPU_NUMS+1) * len(candidate_batch) 增加在两个GPU上都放置模型的动作(因为维度问题，这两个动作batch一致)，则可选择的“GPU”为idx=1，idx=2，以及idx=1和2，因此GPU_NUMS+1
        self.exec_action_length = int(math.pow((1+GPU_NUMS) * len(candidate_batch), MODEL_NUMS))
        self.cache_action_length = int(math.pow(2, GPU_NUMS * MODEL_NUMS))
        self.action_space = spaces.MultiDiscrete([self.exec_action_length, self.cache_action_length])

        # 4模型时
        # observation space: （模型当前时刻请求数 + 模型当前时刻BUFFER队列长度 + 模型在GPU1是否缓存 + GPU2是否缓存 +...+ GPUN是否缓存）* 模型数量 + 两个GPU缓存模型使用的内存量
        # 其中两个15是表示两个GPU的缓存使用量，15是因为有4个模型，缓存可能有15种，c41+c42+c43+c44=4+6+4+1=15
        # 然后的24个800表示的是当前时刻之前的3个时刻的4模型历史输入流，以及当前时刻之后3个时刻的4模型未来输入流（共24个）
        self.observation_length = int((2 + GPU_NUMS) * MODEL_NUMS) + 2 + 12 + 12
        self.observation_space = spaces.MultiDiscrete([800, 800, 800, 800,
                                                       500, 500, 500, 500,
                                                       2, 2, 2, 2, 
                                                       2, 2, 2, 2, 15, 15,
                                                       800, 800, 800, 800,
                                                       800, 800, 800, 800,
                                                       800, 800, 800, 800,
                                                       800, 800, 800, 800,
                                                       800, 800, 800, 800,
                                                       800, 800, 800, 800], dtype=np.float32)

        # 6模型时
        # observation space: （模型当前时刻请求数 + 模型当前时刻BUFFER队列长度 + 模型在GPU1是否缓存 + GPU2是否缓存 +...+ GPUN是否缓存）* 模型数量
        # self.observation_length = int((2 + GPU_NUMS) * MODEL_NUMS)
        # self.observation_space = spaces.MultiDiscrete([800, 800, 800, 800, 800, 800,
        #                                                500, 500, 500, 500, 500, 500,
        #                                                2, 2, 2, 2, 2, 2,
        #                                                2, 2, 2, 2, 2, 2], dtype=np.float32)

        self.model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']                                              
        self.CART_dtr = joblib.load("/home/hpj/project/cleanrl/cleanrl/data/drl/cart.pkl") # 读取
        self.request_retention_time_scale = 3           # 表示请求经过多少个time scale之后没被处理的话就丢弃了
        self.time_scale = 1                             # 代表1s
        self.time_duration = 240
        self.buffer_max_length = 500
        self.evaluation = evaluation_flag
        self.discard_cost_alpha = discard_cost_alpha    # 加载模型时，计算加载损失reward的系数
        self.memory_reward_alpha = memory_reward_alpha
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
        # 初始化evaluation所需的各个字典
        self.defineEvaluationDict()
        # 获取实际动作与动作索引的映射关系
        _= self.getActionIndexMapping()
        
        evaGetRealResult()
        
        # print("gpucluste1a_p_4m: This is {}".format(os.path.abspath(os.path.dirname(__file__))))


    def step_igniter(self, pair_action): 
        # 执行该函数输出所有GPU上的所有place信息
        def showAllPlaceDetails(self):
            for GPU in self.GPUS:
                GPU.showPlaces()

        # 执行该函数则对GPU上的模型缓存状态进行更改
        def changeCacheState(self, gpu_idx, model_name, target_gpu_model_cache_state):
            # 该模型目标的缓存状态是target_cache_state
            # 该模型当前的缓存状态是cur_gpu_model_cache_state
            cur_gpu_model_cache_state = self.cache_state[model_name][gpu_idx]
            load_time = 0

            # 如果当前没缓存，但是目标要缓存，则增加一个load_time
            if cur_gpu_model_cache_state == 0 and target_gpu_model_cache_state == 1:
                # print("changestate: load {} on GPU {}".format(model_name, gpu_idx))
                load_time += self.real_loading_time[model_name]

                # 更改GPU用于缓存模型的内存量
                if self.gpu_cached_memory[gpu_idx] + self.model_memory[model_name] <= self.max_memory_size:
                    self.gpu_cached_memory[gpu_idx] += self.model_memory[model_name]
                else:
                    print("{}, 超出最大缓存上限?".format(self.time))

                # 记录当前模型被load了一次
                self.load_model_times[model_name] += 1
                # 更改缓存状态
                self.cache_state[model_name][gpu_idx] = target_gpu_model_cache_state

            # 如果当前缓存了，但是目标不要缓存，则执行卸载
            elif target_gpu_model_cache_state == 0 and cur_gpu_model_cache_state == 1:
                # print("changestate: unload {} on GPU {}".format(model_name, gpu_idx))
                # 更改GPU用于缓存模型的内存量
                if self.gpu_cached_memory[gpu_idx] - self.model_memory[model_name] >= 0:
                    self.gpu_cached_memory[gpu_idx] -= self.model_memory[model_name]
                else:
                    print("{}, 使用的缓存量小于0了?".format(self.time))

                # 记录当前模型被unload了一次
                self.unload_model_times[model_name] += 1
                # 更改缓存状态
                self.cache_state[model_name][gpu_idx] = target_gpu_model_cache_state
                

            # 其他情况说明目标状态与当前状态一致，不需要做动作
            else:
                pass
            # print("cache state:", self.cache_state)
            # print("memory state:", self.gpu_cached_memory)
            # 如果有发生模型加载，则返回因为load模型而花费的时间
            return load_time
      
        # 执行该函数则根据当前action在各个GPU上放置places
        def placeAccordingToActions(self, exec_action):
            # if len(action) != MODEL_NUMS or len(self.model_names) != MODEL_NUMS:
            #     print("每时刻动作数量与模型种类数不符~")
            #     return
            config = exec_action
            # config: (['resnet50', 1, 8], ['vgg19', 1, 8], ['densenet201', 0, 8], ['mobilenet', 0, 16])

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

        # 执行该函数则检查各个模型实际吞吐量，并根据大小关系增删buffer数值，该函数返回实际执行时间t和load model所带来的cost
        def checkThroughput(self):
            # print("{} checkThroughput".format(self.time))
            def calSingleGPUExecTime(model_name, model_throughput, input_sum, gpu_idx):
                tmp_t = 0
                # 如果当前吞吐量大于输入
                if model_throughput >= input_sum:
                    # buffer清空
                    self.buffers[model_name] = 0
                    # 记录当前执行时间，因为最大执行时间即为avaliable_time[gpu_idx]，因此原来的time_scale改为avaliable_time[gpu_idx]
                    tmp_t += (avaliable_time[gpu_idx] * input_sum) / model_throughput
                    # 记录当前模型当前时刻的执行时间
                    self.exec_time[model_name][self.time] = (self.time_scale * input_sum) / model_throughput
                # 如果当前吞吐量小于输入
                else:
                    # 时间加一个avaliable_time[gpu_idx]
                    tmp_t += avaliable_time[gpu_idx]
                    # 缓存队列长度超出上限也无所谓，每次调度最后会有检查队列长度的函数，将超出的部分丢弃，并返回负reward
                    self.buffers[model_name] = input_sum - model_throughput
                    # 记录当前模型当前时刻的执行时间
                    self.exec_time[model_name][self.time] = self.time_scale
                return tmp_t

            def calDoubleGPUExecTime(model_name, input_sum):
                # 定义该模型在两个GPU上执行的时间
                tmp_t1 = 0
                tmp_t2 = 0

                # 定义一个变量记录两个GPU上没有执行完的请求，最后要加入缓存队列
                remain_request = 0

                # 现在要将这个总的输入分配到两个GPU上执行，分配比例即为两个GPU上的吞吐量比例
                throughput_sum = sum(self.throughput[model_name])

                gpu1_throughput = self.throughput[model_name][0]
                gpu2_throughput = self.throughput[model_name][1]
                gpu1_request_nums = (self.throughput[model_name][0] / throughput_sum) * input_sum
                gpu2_request_nums = (self.throughput[model_name][1] / throughput_sum) * input_sum

                # 如果GPU1上的吞吐量大于了总输入
                if  gpu1_throughput >= gpu1_request_nums:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t1 += (avaliable_time[0] * gpu1_request_nums) / gpu1_throughput
                    # 以及记录当前执行时间，因为还要计算GPU2上的，所以这里记录的时间×0.5（即GPU1 GPU2做平均）
                    self.exec_time[model_name][self.time] = (self.time_scale * gpu1_request_nums) / gpu1_throughput * 0.5
                # 如果没大于总输入
                else:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t1 += avaliable_time[0]
                    # 记录没执行完的剩余请求数
                    remain_request += gpu1_request_nums - gpu1_throughput
                    # 然后记录当前模型当前时刻的执行时间，也要乘0.5
                    self.exec_time[model_name][self.time] = self.time_scale * 0.5

                # 同样的逻辑，GPU2再来一遍
                # 如果GPU2上的吞吐量大于了总输入
                if  gpu2_throughput >= gpu2_request_nums:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t2 += (avaliable_time[1] * gpu2_request_nums) / gpu2_throughput
                    # 以及记录当前执行时间，因为还要计算GPU1上的，所以这里记录的时间×0.5（即GPU1 GPU2做平均）
                    self.exec_time[model_name][self.time] = (self.time_scale * gpu2_request_nums) / gpu2_throughput * 0.5
                # 如果没大于总输入
                else:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t2 += avaliable_time[1]
                    # 记录没执行完的剩余请求数
                    remain_request += gpu2_request_nums - gpu2_throughput
                    # 然后记录当前模型当前时刻的执行时间，也要乘0.5
                    self.exec_time[model_name][self.time] = self.time_scale * 0.5

                # 最后操作缓存=两个GPU上没执行完的请求之和
                self.buffers[model_name] = remain_request

                # 这里返回两个GPU上执行的平均时间
                return (tmp_t1 + tmp_t2) / 2
            
            t = 0
            total_load_time = 0
            
            # 3.24修改，将load模型使用的时间算入计算时间内，如果发生模型加载，则本时刻的可用时间将减少
            # 我们为每个GPU定义实际可执行推理的时长avaliable_time
            avaliable_time = [self.time_scale for _ in range(GPU_NUMS)]

            # 第一次遍历GPU上的place，用于计算模型加载消耗的时间
            for idx, GPU in enumerate(self.GPUS):
                # 定义存储当前GPU上本时刻内因为加载模型而消耗的时间变量
                cur_gpu_load_time = 0
                for place in GPU.places:
                    model_name = place.model_name
                    # t增加load模型带来的cost，如果当前GPU上已有model_name模型，则self.changeCacheState返回0
                    cur_model_load_time = changeCacheState(self, idx, model_name, 1)
                    cur_gpu_load_time += cur_model_load_time
                    total_load_time += cur_model_load_time
                # 内层循环结束，当前GPU将消耗的用于加载模型的时间为load_model_time
                # 因此，avaliable变量内当前GPU可用于推理的时间应对应减少
                avaliable_time[idx] -= cur_gpu_load_time 
                

            # 第二次遍历GPU上的place，用于统计每个模型的吞吐量
            # 每次检查吞吐量前要先重置吞吐量计数器
            self.defineThroughputState()
            for idx, GPU in enumerate(self.GPUS):
                for place in GPU.places:
                    model_name = place.model_name
                    # 因为可能发生模型加载，因此之前计算的1s的吞吐量应该乘以一个比例
                    # 即实际的吞吐量=之前计算的1s内的吞吐量*当前GPU可用于推理的时间，即avaliable_time[idx]
                    self.throughput[model_name][idx] += place.throughput * avaliable_time[idx]

            for model_name in self.model_names:
                model_request_nums = self.input_streams[model_name][self.time]
                model_buffer_length = self.buffers[model_name]
                self.model_throughput[model_name][self.time] = sum(self.throughput[model_name])

                # 首先判断当前模型是否在两个GPU上都执行，如果否则和之前一样的逻辑   
                if self.throughput[model_name][0] == 0 and self.throughput[model_name][1] == 0:
                    print("error, no model cached")
                    break
                # 如果只在GPU1上执行
                elif self.throughput[model_name][0] != 0 and self.throughput[model_name][1] == 0:
                    t += calSingleGPUExecTime(model_name, self.throughput[model_name][0], model_request_nums + model_buffer_length, 0)
                # 如果只在GPU2上执行
                elif self.throughput[model_name][0] == 0 and self.throughput[model_name][1] != 0:
                    t += calSingleGPUExecTime(model_name, self.throughput[model_name][1], model_request_nums + model_buffer_length, 1)
                # 如果在两个GPU上都执行
                elif self.throughput[model_name][0] != 0 and self.throughput[model_name][1] != 0:
                    t += calDoubleGPUExecTime(model_name, model_request_nums + model_buffer_length)

            return t, total_load_time
            

            # for model_name in self.model_names:
            #     # 获取当前模型的请求数、可达到的吞吐量、缓存队列长度
            #     model_request_nums = self.input_streams[model_name][self.time]
            #     model_throughput = self.throughput[model_name]
            #     model_buffer_length = self.buffers[model_name]
            #     # 如果吞吐量大于等于缓存队列长度，则缓存清0
            #     if model_throughput >= model_buffer_length:
            #         model_throughput -= model_buffer_length
            #         self.buffers[model_name] = 0
            #         # 如果吞吐量-缓存队列长度后 仍大于等于 请求到达数，则说明当前时刻可完全处理，t增加处理的时间
            #         if model_throughput >= model_request_nums:
            #             t += (self.time_scale * model_throughput) / (model_request_nums + model_buffer_length)
            #             # 记录当前模型当前时刻的执行时间
            #             self.exec_time[model_name][self.time] = (self.time_scale * model_throughput) / (model_request_nums + model_buffer_length)
            #         # 如果吞吐量-缓存队列长度后 小于 请求到达数，则说明有一部分请求不能被处理，定义remaining_request_nums表示这部分请求
            #         # 并将这部分请求存入buffer，t增加一个时间尺度timescale
            #         elif model_throughput < model_request_nums:
            #             remaining_request_nums = model_request_nums - model_throughput
            #             self.buffers[model_name] += remaining_request_nums
            #             t += self.time_scale
            #             # 记录当前模型当前时刻的执行时间
            #             self.exec_time[model_name][self.time] = self.time_scale
            #     # 如果吞吐量小于缓存队列长度，则缓存队列长度减少吞吐量大小，时间增加t，然后缓存队列再增加当前请求到达数个请求
            #     elif model_throughput < model_buffer_length:
            #         self.buffers[model_name] -= model_throughput
            #         t += self.time_scale
            #         # 记录当前模型当前时刻的执行时间
            #         self.exec_time[model_name][self.time] = self.time_scale
            #         self.buffers[model_name] += model_request_nums

            
        # 执行该函数则清空所有GPU上的place信息，这个函数应该在每次step结束之前调用且必须调用
        def clearGPUPlaces(self):
            for GPU in self.GPUS:
                GPU.deletePlaces()
        
        # 执行该函数则根据cache_action对各个模型进行卸载，cache_action中为0的位置表示对应模型会被卸载
        def unloadModelAccordingToCacheAction(self, cache_action):
            # print("{}, unloadModelxx".format(self.time))
            #cache_action_list是[0,0,1,1,1,1,0,0]这样长度为8的list，如果哪个位置为0，我就卸载哪个位置的模型
            cache_action_list = self.cache_action_config_map[cache_action]

            for gpu_idx in range(GPU_NUMS):
                for model_idx, model_name in enumerate(self.model_names):
                    # 获取当前判断的模型在cache_action_list中的位置
                    cur_idx = gpu_idx * GPU_NUMS + model_idx
                    if cache_action_list[cur_idx] == 0:
                        # 第三个参数0代表卸载，卸载时changeCacheState的返回值(加载带来的cost)一定为0，这里选择不接收，因为用不到
                        _ = changeCacheState(self, gpu_idx, model_name, 0)

            print(self.cache_state)
            # for循环结束就完成了cache_action中表示的所有模型的卸载动作，无需返回值

        # 执行该函数则根据当前两个GPU上模型缓存的占用空间获得reward，占用的越少获得reward越大
        def getCacheReward(self):
            # 用total_cached_memory记录当前缓存模型而使用的内存大小
            gpu1_cached_memory = self.gpu_cached_memory[0]
            gpu2_cached_memory = self.gpu_cached_memory[1]
            total_cached_memory = gpu1_cached_memory + gpu2_cached_memory

            # 记录当前时刻每个GPU上（最大缓存模型使用空间-当前使用空间）的差，表示模型卸载腾出了多少有效空间
            self.gpu_available_cache_memory_trace[self.time][0] = self.max_memory_size - gpu1_cached_memory
            self.gpu_available_cache_memory_trace[self.time][1] = self.max_memory_size - gpu2_cached_memory

            # 返回当前缓存所使用的空间与缓存上限的内存空间之差，表示因为模型卸载而释放的内存大小
            return self.max_memory_size - total_cached_memory

        # 执行该函数检查所有缓存队列长度，如果超过上限则丢弃，并返回负reward
        def checkBufferLength(self):
            discard_nums = 0
            for model_name in self.model_names:
                # 如果buffer长度超过上限
                if self.buffers[model_name] > self.buffer_max_length:
                    # discard_nums增加当前丢弃的请求数量
                    discard_nums += self.buffers[model_name] - self.buffer_max_length
                    # 记录当前模型丢弃的请求数量
                    self.discard_request_nums[model_name][self.time] += self.buffers[model_name] - self.buffer_max_length
                    # buffer丢弃多余的，剩下最大上限数量的请求
                    self.buffers[model_name] = self.buffer_max_length

            return discard_nums

        # ---------------- 函数执行部分 -----------------
        exec_action, cache_action = pair_action[0], int(pair_action[1])
        done = False
        reward = 0
        # 先根据cache_action执行卸载，此时不获得任何reward，只更改cache状态、更改GPU用来缓存模型的内存大小
        unloadModelAccordingToCacheAction(self, cache_action)

        # 再根据exec_action执行推理，首先放置place
        placeAccordingToActions(self, exec_action)

        # 再根据干扰计算各个place的吞吐量，保存在place里
        getRealThroughput(self)

        # 根据吞吐量计算当前每个模型执行时间，该部分将有部分模型被重新缓存到GPU上
        exec_time, load_time = checkThroughput(self)

        # 根据当前用于缓存模型内存的空闲部分，（即self.max_memory_size-当前缓存使用内存大小），获得reward，空闲越多获得reward越大
        reward += self.memory_reward_alpha * getCacheReward(self)
    
        reward -= (exec_time + load_time)
        reward -= self.discard_cost_alpha * checkBufferLength(self)

        clearGPUPlaces(self)
        
        next_state = np.array([])
        if self.time + 1 < self.time_duration:
            next_state = np.array(self.getStateFunc(self.time + 1))
        else:
            done = True
        self.time += 1

        resnet50_dis_nums = sum(self.discard_request_nums['resnet50'])
        resnet50_dis_percent = resnet50_dis_nums / self.input_streams_sum['resnet50']
        resnet50_load_times = self.load_model_times['resnet50']
        resnet50_unload_times = self.unload_model_times['resnet50']
        resnet50_exec_time = sum(self.exec_time['resnet50'])

        vgg19_dis_nums = sum(self.discard_request_nums['vgg19'])
        vgg19_dis_percent = vgg19_dis_nums / self.input_streams_sum['vgg19']
        vgg19_load_times = self.load_model_times['vgg19']
        vgg19_unload_times = self.unload_model_times['vgg19']
        vgg19_exec_time = sum(self.exec_time['vgg19'])

        densenet201_dis_nums = sum(self.discard_request_nums['densenet201'])
        densenet201_dis_percent = densenet201_dis_nums / self.input_streams_sum['densenet201']
        densenet201_load_times = self.load_model_times['densenet201']
        densenet201_unload_times = self.unload_model_times['densenet201']
        densenet201_exec_time = sum(self.exec_time['densenet201'])

        mobilenet_dis_nums = sum(self.discard_request_nums['mobilenet'])
        mobilenet_dis_percent = mobilenet_dis_nums / self.input_streams_sum['mobilenet']
        mobilenet_load_times = self.load_model_times['mobilenet']
        mobilenet_unload_times = self.unload_model_times['mobilenet']
        mobilenet_exec_time = sum(self.exec_time['mobilenet'])

        gpu1_cache_memory_free = 0
        gpu2_cache_memory_free = 0
        total_cache_memory_free = 0
        # 统计当前时间序列中，每个时刻每个GPU上（最大缓存模型使用空间-当前缓存模型使用空间）的值
        # 外层循环遍历时间t
        for time, memory_line in enumerate(self.gpu_available_cache_memory_trace):
            # 内层遍历两个GPU
            gpu1_cache_memory_free += memory_line[0] / 1024 / self.time_duration
            gpu2_cache_memory_free += memory_line[1] / 1024 / self.time_duration
        total_cache_memory_free = gpu1_cache_memory_free + gpu2_cache_memory_free


        # 当前模型执行总时间，用这个指标观察reward中有关执行时间的变化
        total_exec_time = resnet50_exec_time + vgg19_exec_time + densenet201_exec_time + mobilenet_exec_time

        if self.evaluation and done:
            extra_message = {
                # dis percent和loadtimes 在每一轮迭代中都会计算，但此处返回的是最后一轮迭代得到的这两个值
                # 按照定义，dis_percent此时使用的是所有时刻t的dis nums之和除以总对应的模型请求到达数量，代表了整个时序流中的dis情况
                # 而loadtimes是每一轮迭代累加，在最后一轮将得到整个时序流中每个模型总的loadtimes数量，因此都符合evaluate标准
                'resnet50_dis_nums':self.discard_request_nums['resnet50'],
                'vgg19_dis_nums':self.discard_request_nums['vgg19'],
                'densenet201_dis_nums':self.discard_request_nums['densenet201'],
                'mobilenet_dis_nums':self.discard_request_nums['mobilenet'],

                'resnet50_max_throughput':self.model_throughput['resnet50'],
                'vgg19_max_throughput':self.model_throughput['vgg19'],
                'densenet201_max_throughput':self.model_throughput['densenet201'],
                'mobilenet_max_throughput':self.model_throughput['mobilenet'],

                'resnet50_exec_time':self.exec_time['resnet50'],
                'vgg19_exec_time':self.exec_time['vgg19'],
                'densenet201_exec_time':self.exec_time['densenet201'],
                'mobilenet_exec_time':self.exec_time['mobilenet'],

                'gpu_available_cache_memory_trace':self.gpu_available_cache_memory_trace,

            }
            return  next_state, reward, done, extra_message
        
        return next_state, reward, done, {'time': self.time - 1, 
                                          "resnet50_dis_nums": resnet50_dis_nums, "resnet50_dis_percent": resnet50_dis_percent,
                                          "resnet50_load_times": resnet50_load_times, "resnet50_unload_times":resnet50_unload_times,
                                          "resnet50_exec_time": resnet50_exec_time,
                                          "vgg19_dis_nums": vgg19_dis_nums, "vgg19_dis_percent": vgg19_dis_percent,
                                          "vgg19_load_times": vgg19_load_times, "vgg19_unload_times": vgg19_unload_times,
                                          "vgg19_exec_time": vgg19_exec_time,
                                          "densenet201_dis_nums": densenet201_dis_nums, "densenet201_dis_percent": densenet201_dis_percent, 
                                          "densenet201_load_times": densenet201_load_times, "densenet201_unload_times": densenet201_unload_times,
                                          "densenet201_exec_time": densenet201_exec_time,
                                          "mobilenet_dis_nums": mobilenet_dis_nums, "mobilenet_dis_percent": mobilenet_dis_percent, 
                                          "mobilenet_load_times": mobilenet_load_times, "mobilenet_unload_times": mobilenet_unload_times,
                                          "mobilenet_exec_time": mobilenet_exec_time,
                                          "total_exec_time": total_exec_time,
                                          "gpu1_cache_memory_free": gpu1_cache_memory_free, "gpu2_cache_memory_free": gpu2_cache_memory_free,
                                          "total_cache_memory_free": total_cache_memory_free}


    def step(self, pair_action): 
        # 执行该函数输出所有GPU上的所有place信息
        def showAllPlaceDetails(self):
            for GPU in self.GPUS:
                GPU.showPlaces()

        # 执行该函数则对GPU上的模型缓存状态进行更改
        def changeCacheState(self, gpu_idx, model_name, target_gpu_model_cache_state):
            # 该模型目标的缓存状态是target_cache_state
            # 该模型当前的缓存状态是cur_gpu_model_cache_state
            cur_gpu_model_cache_state = self.cache_state[model_name][gpu_idx]
            load_time = 0

            # 如果当前没缓存，但是目标要缓存，则增加一个load_time
            if cur_gpu_model_cache_state == 0 and target_gpu_model_cache_state == 1:
                # print("changestate: load {} on GPU {}".format(model_name, gpu_idx))
                load_time += self.real_loading_time[model_name]

                # 更改GPU用于缓存模型的内存量
                if self.gpu_cached_memory[gpu_idx] + self.model_memory[model_name] <= self.max_memory_size:
                    self.gpu_cached_memory[gpu_idx] += self.model_memory[model_name]
                else:
                    print("{}, 超出最大缓存上限?".format(self.time))

                # 记录当前模型被load了一次
                self.load_model_times[model_name] += 1
                # 更改缓存状态
                self.cache_state[model_name][gpu_idx] = target_gpu_model_cache_state

            # 如果当前缓存了，但是目标不要缓存，则执行卸载
            elif target_gpu_model_cache_state == 0 and cur_gpu_model_cache_state == 1:
                # print("changestate: unload {} on GPU {}".format(model_name, gpu_idx))
                # 更改GPU用于缓存模型的内存量
                if self.gpu_cached_memory[gpu_idx] - self.model_memory[model_name] >= 0:
                    self.gpu_cached_memory[gpu_idx] -= self.model_memory[model_name]
                else:
                    print("{}, 使用的缓存量小于0了?".format(self.time))

                # 记录当前模型被unload了一次
                self.unload_model_times[model_name] += 1
                # 更改缓存状态
                self.cache_state[model_name][gpu_idx] = target_gpu_model_cache_state
                

            # 其他情况说明目标状态与当前状态一致，不需要做动作
            else:
                pass
            # print("cache state:", self.cache_state)
            # print("memory state:", self.gpu_cached_memory)
            # 如果有发生模型加载，则返回因为load模型而花费的时间
            return load_time
      
        # 执行该函数则根据当前action在各个GPU上放置places
        def placeAccordingToActions(self, exec_action):
            # if len(action) != MODEL_NUMS or len(self.model_names) != MODEL_NUMS:
            #     print("每时刻动作数量与模型种类数不符~")
            #     return
            config = self.exec_action_config_map[exec_action]
            print(config)
            # config: (['resnet50', 1, 8], ['vgg19', 1, 8], ['densenet201', 0, 8], ['mobilenet', 0, 16])

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

        # 执行该函数则检查各个模型实际吞吐量，并根据大小关系增删buffer数值，该函数返回实际执行时间t和load model所带来的cost
        def checkThroughput(self):
            # print("{} checkThroughput".format(self.time))
            def calSingleGPUExecTime(model_name, model_throughput, input_sum, gpu_idx):
                tmp_t = 0
                # 如果当前吞吐量大于输入
                if model_throughput >= input_sum:
                    # buffer清空
                    self.buffers[model_name] = 0
                    # 记录当前执行时间，因为最大执行时间即为avaliable_time[gpu_idx]，因此原来的time_scale改为avaliable_time[gpu_idx]
                    tmp_t += (avaliable_time[gpu_idx] * input_sum) / model_throughput
                    # 记录当前模型当前时刻的执行时间
                    self.exec_time[model_name][self.time] = (self.time_scale * input_sum) / model_throughput
                # 如果当前吞吐量小于输入
                else:
                    # 时间加一个avaliable_time[gpu_idx]
                    tmp_t += avaliable_time[gpu_idx]
                    # 缓存队列长度超出上限也无所谓，每次调度最后会有检查队列长度的函数，将超出的部分丢弃，并返回负reward
                    self.buffers[model_name] = input_sum - model_throughput
                    # 记录当前模型当前时刻的执行时间
                    self.exec_time[model_name][self.time] = self.time_scale
                return tmp_t

            def calDoubleGPUExecTime(model_name, input_sum):
                # 定义该模型在两个GPU上执行的时间
                tmp_t1 = 0
                tmp_t2 = 0

                # 定义一个变量记录两个GPU上没有执行完的请求，最后要加入缓存队列
                remain_request = 0

                # 现在要将这个总的输入分配到两个GPU上执行，分配比例即为两个GPU上的吞吐量比例
                throughput_sum = sum(self.throughput[model_name])

                gpu1_throughput = self.throughput[model_name][0]
                gpu2_throughput = self.throughput[model_name][1]
                gpu1_request_nums = (self.throughput[model_name][0] / throughput_sum) * input_sum
                gpu2_request_nums = (self.throughput[model_name][1] / throughput_sum) * input_sum

                # 如果GPU1上的吞吐量大于了总输入
                if  gpu1_throughput >= gpu1_request_nums:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t1 += (avaliable_time[0] * gpu1_request_nums) / gpu1_throughput
                    # 以及记录当前执行时间，因为还要计算GPU2上的，所以这里记录的时间×0.5（即GPU1 GPU2做平均）
                    self.exec_time[model_name][self.time] = (self.time_scale * gpu1_request_nums) / gpu1_throughput * 0.5
                # 如果没大于总输入
                else:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t1 += avaliable_time[0]
                    # 记录没执行完的剩余请求数
                    remain_request += gpu1_request_nums - gpu1_throughput
                    # 然后记录当前模型当前时刻的执行时间，也要乘0.5
                    self.exec_time[model_name][self.time] = self.time_scale * 0.5

                # 同样的逻辑，GPU2再来一遍
                # 如果GPU2上的吞吐量大于了总输入
                if  gpu2_throughput >= gpu2_request_nums:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t2 += (avaliable_time[1] * gpu2_request_nums) / gpu2_throughput
                    # 以及记录当前执行时间，因为还要计算GPU1上的，所以这里记录的时间×0.5（即GPU1 GPU2做平均）
                    self.exec_time[model_name][self.time] = (self.time_scale * gpu2_request_nums) / gpu2_throughput * 0.5
                # 如果没大于总输入
                else:
                    # 则时间tmp_t1加上执行所用时间
                    tmp_t2 += avaliable_time[1]
                    # 记录没执行完的剩余请求数
                    remain_request += gpu2_request_nums - gpu2_throughput
                    # 然后记录当前模型当前时刻的执行时间，也要乘0.5
                    self.exec_time[model_name][self.time] = self.time_scale * 0.5

                # 最后操作缓存=两个GPU上没执行完的请求之和
                self.buffers[model_name] = remain_request

                # 这里返回两个GPU上执行的平均时间
                return (tmp_t1 + tmp_t2) / 2
            
            t = 0
            total_load_time = 0
            
            # 3.24修改，将load模型使用的时间算入计算时间内，如果发生模型加载，则本时刻的可用时间将减少
            # 我们为每个GPU定义实际可执行推理的时长avaliable_time
            avaliable_time = [self.time_scale for _ in range(GPU_NUMS)]

            # 第一次遍历GPU上的place，用于计算模型加载消耗的时间
            for idx, GPU in enumerate(self.GPUS):
                # 定义存储当前GPU上本时刻内因为加载模型而消耗的时间变量
                cur_gpu_load_time = 0
                for place in GPU.places:
                    model_name = place.model_name
                    # t增加load模型带来的cost，如果当前GPU上已有model_name模型，则self.changeCacheState返回0
                    cur_model_load_time = changeCacheState(self, idx, model_name, 1)
                    cur_gpu_load_time += cur_model_load_time
                    total_load_time += cur_model_load_time
                # 内层循环结束，当前GPU将消耗的用于加载模型的时间为load_model_time
                # 因此，avaliable变量内当前GPU可用于推理的时间应对应减少
                avaliable_time[idx] -= cur_gpu_load_time 
                

            # 第二次遍历GPU上的place，用于统计每个模型的吞吐量
            # 每次检查吞吐量前要先重置吞吐量计数器
            self.defineThroughputState()
            # for idx, GPU in enumerate(self.GPUS):
            #     for place in GPU.places:
            #         model_name = place.model_name
            #         # 因为可能发生模型加载，因此之前计算的1s的吞吐量应该乘以一个比例
            #         # 即实际的吞吐量=之前计算的1s内的吞吐量*当前GPU可用于推理的时间，即avaliable_time[idx]
            #         self.throughput[model_name][idx] += place.throughput * avaliable_time[idx]
            for model_name in self.model_names:
                print("111")
                self.throughput[model_name][0] = self.evaRealThroughput[model_name][self.time][0]
                self.throughput[model_name][1] = self.evaRealThroughput[model_name][self.time][1]


            for model_name in self.model_names:
                model_request_nums = self.input_streams[model_name][self.time]
                model_buffer_length = self.buffers[model_name]
                self.model_throughput[model_name][self.time] = sum(self.throughput[model_name])

                # 首先判断当前模型是否在两个GPU上都执行，如果否则和之前一样的逻辑   
                if self.throughput[model_name][0] == 0 and self.throughput[model_name][1] == 0:
                    print("error, no model cached")
                    break
                # 如果只在GPU1上执行
                elif self.throughput[model_name][0] != 0 and self.throughput[model_name][1] == 0:
                    t += calSingleGPUExecTime(model_name, self.throughput[model_name][0], model_request_nums + model_buffer_length, 0)
                # 如果只在GPU2上执行
                elif self.throughput[model_name][0] == 0 and self.throughput[model_name][1] != 0:
                    t += calSingleGPUExecTime(model_name, self.throughput[model_name][1], model_request_nums + model_buffer_length, 1)
                # 如果在两个GPU上都执行
                elif self.throughput[model_name][0] != 0 and self.throughput[model_name][1] != 0:
                    t += calDoubleGPUExecTime(model_name, model_request_nums + model_buffer_length)

            return t, total_load_time

            
        # 执行该函数则清空所有GPU上的place信息，这个函数应该在每次step结束之前调用且必须调用
        def clearGPUPlaces(self):
            for GPU in self.GPUS:
                GPU.deletePlaces()
        
        # 执行该函数则根据cache_action对各个模型进行卸载，cache_action中为0的位置表示对应模型会被卸载
        def unloadModelAccordingToCacheAction(self, cache_action):
            # print("{}, unloadModelxx".format(self.time))
            #cache_action_list是[0,0,1,1,1,1,0,0]这样长度为8的list，如果哪个位置为0，我就卸载哪个位置的模型
            cache_action_list = self.cache_action_config_map[cache_action]
            print(cache_action, cache_action_list)
            print(self.cache_state)
            for gpu_idx in range(GPU_NUMS):
                for model_idx, model_name in enumerate(self.model_names):
                    # 获取当前判断的模型在cache_action_list中的位置
                    cur_idx = gpu_idx * MODEL_NUMS + model_idx
                    if cache_action_list[cur_idx] == 0:
                        # 第三个参数0代表卸载，卸载时changeCacheState的返回值(加载带来的cost)一定为0，这里选择不接收，因为用不到
                        _ = changeCacheState(self, gpu_idx, model_name, 0)
            print(self.cache_state)
            # for循环结束就完成了cache_action中表示的所有模型的卸载动作，无需返回值

        # 执行该函数则根据当前两个GPU上模型缓存的占用空间获得reward，占用的越少获得reward越大
        def getCacheReward(self):
            # 用total_cached_memory记录当前缓存模型而使用的内存大小
            gpu1_cached_memory = self.gpu_cached_memory[0]
            gpu2_cached_memory = self.gpu_cached_memory[1]
            total_cached_memory = gpu1_cached_memory + gpu2_cached_memory

            # 记录当前时刻每个GPU上（最大缓存模型使用空间-当前使用空间）的差，表示模型卸载腾出了多少有效空间
            self.gpu_available_cache_memory_trace[self.time][0] = self.max_memory_size - gpu1_cached_memory
            self.gpu_available_cache_memory_trace[self.time][1] = self.max_memory_size - gpu2_cached_memory

            # 返回当前缓存所使用的空间与缓存上限的内存空间之差，表示因为模型卸载而释放的内存大小
            return self.max_memory_size - total_cached_memory

        # 执行该函数检查所有缓存队列长度，如果超过上限则丢弃，并返回负reward
        def checkBufferLength(self):
            discard_nums = 0
            for model_name in self.model_names:
                # 如果buffer长度超过上限
                if self.buffers[model_name] > self.buffer_max_length:
                    # discard_nums增加当前丢弃的请求数量
                    discard_nums += self.buffers[model_name] - self.buffer_max_length
                    # 记录当前模型丢弃的请求数量
                    self.discard_request_nums[model_name][self.time] += self.buffers[model_name] - self.buffer_max_length
                    # buffer丢弃多余的，剩下最大上限数量的请求
                    self.buffers[model_name] = self.buffer_max_length

            return discard_nums

        # ---------------- 函数执行部分 -----------------
        exec_action, cache_action = int(pair_action[0]), int(pair_action[1])
        done = False
        reward = 0
        # 先根据cache_action执行卸载，此时不获得任何reward，只更改cache状态、更改GPU用来缓存模型的内存大小
        unloadModelAccordingToCacheAction(self, cache_action)

        # 再根据exec_action执行推理，首先放置place
        placeAccordingToActions(self, exec_action)
        print("---------------")
        # 再根据干扰计算各个place的吞吐量，保存在place里
        getRealThroughput(self)

        # 根据吞吐量计算当前每个模型执行时间，该部分将有部分模型被重新缓存到GPU上
        exec_time, load_time = checkThroughput(self)

        # 根据当前用于缓存模型内存的空闲部分，（即self.max_memory_size-当前缓存使用内存大小），获得reward，空闲越多获得reward越大
        reward += self.memory_reward_alpha * getCacheReward(self)
    
        reward -= (exec_time + load_time)
        reward -= self.discard_cost_alpha * checkBufferLength(self)

        clearGPUPlaces(self)
        
        next_state = np.array([])
        if self.time + 1 < self.time_duration:
            next_state = np.array(self.getStateFunc(self.time + 1))
        else:
            done = True
        self.time += 1

        resnet50_dis_nums = sum(self.discard_request_nums['resnet50'])
        resnet50_dis_percent = resnet50_dis_nums / self.input_streams_sum['resnet50']
        resnet50_load_times = self.load_model_times['resnet50']
        resnet50_unload_times = self.unload_model_times['resnet50']
        resnet50_exec_time = sum(self.exec_time['resnet50'])

        vgg19_dis_nums = sum(self.discard_request_nums['vgg19'])
        vgg19_dis_percent = vgg19_dis_nums / self.input_streams_sum['vgg19']
        vgg19_load_times = self.load_model_times['vgg19']
        vgg19_unload_times = self.unload_model_times['vgg19']
        vgg19_exec_time = sum(self.exec_time['vgg19'])

        densenet201_dis_nums = sum(self.discard_request_nums['densenet201'])
        densenet201_dis_percent = densenet201_dis_nums / self.input_streams_sum['densenet201']
        densenet201_load_times = self.load_model_times['densenet201']
        densenet201_unload_times = self.unload_model_times['densenet201']
        densenet201_exec_time = sum(self.exec_time['densenet201'])

        mobilenet_dis_nums = sum(self.discard_request_nums['mobilenet'])
        mobilenet_dis_percent = mobilenet_dis_nums / self.input_streams_sum['mobilenet']
        mobilenet_load_times = self.load_model_times['mobilenet']
        mobilenet_unload_times = self.unload_model_times['mobilenet']
        mobilenet_exec_time = sum(self.exec_time['mobilenet'])

        gpu1_cache_memory_free = 0
        gpu2_cache_memory_free = 0
        total_cache_memory_free = 0
        # 统计当前时间序列中，每个时刻每个GPU上（最大缓存模型使用空间-当前缓存模型使用空间）的值
        # 外层循环遍历时间t
        for time, memory_line in enumerate(self.gpu_available_cache_memory_trace):
            # 内层遍历两个GPU
            gpu1_cache_memory_free += memory_line[0] / 1024 / self.time_duration
            gpu2_cache_memory_free += memory_line[1] / 1024 / self.time_duration
        total_cache_memory_free = gpu1_cache_memory_free + gpu2_cache_memory_free


        # 当前模型执行总时间，用这个指标观察reward中有关执行时间的变化
        total_exec_time = resnet50_exec_time + vgg19_exec_time + densenet201_exec_time + mobilenet_exec_time

        if self.evaluation and done:
            extra_message = {
                # dis percent和loadtimes 在每一轮迭代中都会计算，但此处返回的是最后一轮迭代得到的这两个值
                # 按照定义，dis_percent此时使用的是所有时刻t的dis nums之和除以总对应的模型请求到达数量，代表了整个时序流中的dis情况
                # 而loadtimes是每一轮迭代累加，在最后一轮将得到整个时序流中每个模型总的loadtimes数量，因此都符合evaluate标准
                'resnet50_dis_nums':self.discard_request_nums['resnet50'],
                'vgg19_dis_nums':self.discard_request_nums['vgg19'],
                'densenet201_dis_nums':self.discard_request_nums['densenet201'],
                'mobilenet_dis_nums':self.discard_request_nums['mobilenet'],

                'resnet50_max_throughput':self.model_throughput['resnet50'],
                'vgg19_max_throughput':self.model_throughput['vgg19'],
                'densenet201_max_throughput':self.model_throughput['densenet201'],
                'mobilenet_max_throughput':self.model_throughput['mobilenet'],

                'resnet50_exec_time':self.exec_time['resnet50'],
                'vgg19_exec_time':self.exec_time['vgg19'],
                'densenet201_exec_time':self.exec_time['densenet201'],
                'mobilenet_exec_time':self.exec_time['mobilenet'],

                'gpu_available_cache_memory_trace':self.gpu_available_cache_memory_trace,
                "total_cache_memory_free": total_cache_memory_free,

            }
            return  next_state, reward, done, extra_message
        
        return next_state, reward, done, {'time': self.time - 1, 
                                          "resnet50_dis_nums": resnet50_dis_nums, "resnet50_dis_percent": resnet50_dis_percent,
                                          "resnet50_load_times": resnet50_load_times, "resnet50_unload_times":resnet50_unload_times,
                                          "resnet50_exec_time": resnet50_exec_time,
                                          "vgg19_dis_nums": vgg19_dis_nums, "vgg19_dis_percent": vgg19_dis_percent,
                                          "vgg19_load_times": vgg19_load_times, "vgg19_unload_times": vgg19_unload_times,
                                          "vgg19_exec_time": vgg19_exec_time,
                                          "densenet201_dis_nums": densenet201_dis_nums, "densenet201_dis_percent": densenet201_dis_percent, 
                                          "densenet201_load_times": densenet201_load_times, "densenet201_unload_times": densenet201_unload_times,
                                          "densenet201_exec_time": densenet201_exec_time,
                                          "mobilenet_dis_nums": mobilenet_dis_nums, "mobilenet_dis_percent": mobilenet_dis_percent, 
                                          "mobilenet_load_times": mobilenet_load_times, "mobilenet_unload_times": mobilenet_unload_times,
                                          "mobilenet_exec_time": mobilenet_exec_time,
                                          "total_exec_time": total_exec_time,
                                          "gpu1_cache_memory_free": gpu1_cache_memory_free, "gpu2_cache_memory_free": gpu2_cache_memory_free,
                                          "total_cache_memory_free": total_cache_memory_free}

    def reset(self):
        # 初始化部分需要对__init__函数中定义的诸多变量进行重置
        self.defineBuffers()
        self.defineInputStream()
        self.defineModelCacheState()
        self.defineDiscardRequest()
        self.defineEvaluationDict()
        self.GPUS = [GPU(i) for i in range(GPU_NUMS)]
        self.time = 0    

        # 获取初始状态
        origin_state = self.getStateFunc(0)
        return np.array(origin_state, dtype=np.float32)

# if __name__ == "__main__":
#     env = GPUEnv1a_mh_p_4m_2g()
#     env.reset()
#     next_s, reward, done, info = env.step((400, 255))
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step1 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # action = 1200
    # env.step((1000,200))
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step2 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # action = 5321
    # env.step(action)
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step3 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # action = 1100
    # env.step(action)
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step4 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # action = 3201
    # env.step(action)
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=step5 over-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
