import os
import time
from itertools import combinations
import csv
import torch

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

    def showDetails(self):
        print("--------------")
        print(self.model_name)
        print("batch:",self.b)
        print("gpu r:",self.gpu_resource)
        print("gpu idx:",self.gpu_idx)

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


model_load_time = {
    'resnet50':0.05,
    'vgg19':0.13,
    'densenet201':0.1,
    'mobilenet':0.025
}

model_id_hash = {
    "resnet50" : 0,
    "vgg19": 1,
    "densenet201" : 2,
    "mobilenet": 3
}

model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']    
candidate_batch = [4, 8, 16] 
GPU_NUMS = 2
GPUS = [GPU(i) for i in range(GPU_NUMS)]
MODEL_NUMS = 4
gpu_states = [0 for i in range(GPU_NUMS * MODEL_NUMS)]
time_duration = 240
exec_action_config_map = dict()
cache_action_config_map = dict()


def getCacheActionIndexMapping():
    for i in range(256):
        binary = list(bin(i)[2:])
        while len(binary) < 8:
            binary.insert(0,0)
        for idx, str in enumerate(binary):
            binary[idx] = int(str)
        cache_action_config_map[i] = binary

# 执行该函数获取动作和索引之间的映射
def getExecActionIndexMapping():
    single_model_config = []
    for m_idx in range(len(model_names)):
        for g_idx in range(GPU_NUMS + 1):     # 3.14修改：增加在两个GPU上都放置模型的动作，因此g_idx循环时增加1，表示两个GPU都放置模型的动作
            for b_idx in range(len(candidate_batch)):
                single_model_config.append((model_names[m_idx], g_idx, candidate_batch[b_idx]))
    tmp_res = list(combinations(single_model_config, MODEL_NUMS))

    cnt = 0
    for full_config in tmp_res:
        # print(full_config)
        name_set = set()
        for config in full_config:
            name_set.add(config[0])
        if len(name_set) == MODEL_NUMS:
            exec_action_config_map[cnt] = full_config
            cnt += 1


def process_cache_action(cache_action):
    def unload_models():
        cache_action_list = cache_action_config_map[cache_action]
        for i in range(len(cache_action_list)):
            if cache_action_list[i] == 0:
                gpu_states[i] = 0
    unload_models()


def process_exec_action(exec_action):
    config = exec_action_config_map[exec_action]
    # config: (['resnet50', 1, 8], ['vgg19', 1, 8], ['densenet201', 0, 8], ['mobilenet', 0, 16])

    total_load_model_time = [0 for _ in range(GPU_NUMS)]

    tmp_gpus = [[] for _ in range(GPU_NUMS)]
    for i in range(len(config)):
        cur_model_name = config[i][0]
        cur_model_gpu_idx = config[i][1]
        cur_model_batch = config[i][2]
        # 如果cur_model_gpu_idx=2说明动作要把该配置放在两个GPU上，则在两个GPU上都增加place
        if cur_model_gpu_idx == 2:
            tmp_gpus[0].append((model_names[i], cur_model_batch))
            tmp_gpus[1].append((model_names[i], cur_model_batch))
        else:
            tmp_gpus[cur_model_gpu_idx].append((model_names[i], cur_model_batch))
    print(tmp_gpus)
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
            GPUS[gpu_idx].addPlaces(new_model_place)

            cur_place_cache_idx = gpu_idx * MODEL_NUMS + model_names.index(cur_model_name)
            if gpu_states[cur_place_cache_idx] == 0:
                gpu_states[cur_place_cache_idx] = 1
                total_load_model_time[gpu_idx] += model_load_time[cur_model_name]
    return total_load_model_time


if __name__ == "__main__":

    getExecActionIndexMapping()
    getCacheActionIndexMapping()
    # print(exec_action_config_map[3882])
    # print(cache_action_config_map[251])


    action_list = []
    with open("./alpha_0.02_base290_action_list.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            action_list.append(row)
    

    gpu1_exec_time = [0 for _ in range(time_duration)]
    gpu2_exec_time = [0 for _ in range(time_duration)]

    # 实际执行
    for t, action in enumerate(action_list):
        print("---------------------------------------")
        for GPU in GPUS:
            GPU.deletePlaces()


        exec_action = int(action[0])
        cache_action = int(action[1])
        # exec_action = int(3882)
        # cache_action = int(251)

        # 首先执行卸载，这部分不产生时间消耗
        process_cache_action(cache_action)

        # 然后进行模型加载，并将place信息保存入GPUS变量
        g1_load_time, g2_load_time = process_exec_action(exec_action)

        g1_exec_time = 1 - g1_load_time
        g2_exec_time = 1 - g2_load_time
        
        # gpu1_exec_time[t] = min(g1_exec_time + 0.1, 1)
        # gpu2_exec_time[t] = min(g2_exec_time + 0.1, 1)
        gpu1_exec_time[t] = g1_exec_time
        gpu2_exec_time[t] = g2_exec_time
    
    with open("gpu_exec_time.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for i in range(time_duration):
            writer.writerow([gpu1_exec_time[i], gpu2_exec_time[i]])
        


        

     





