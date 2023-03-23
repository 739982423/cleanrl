# 最小负载调度主程序
import csv
import random
import gym
import collections
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime as dt

GPU_NUMS = 2

class minWorkloadScheduler():
    def __init__(self, env) -> None:
        self.index_action_map = env.getActionIndexMapping()
        self.action_index_map = collections.defaultdict(int)
        self.getActionIndexMap()
        self.batch = 16
        self.model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']
        self.combinations = list(itertools.permutations(self.model_names, len(self.model_names)))
        self.gpu_state = [0 for _ in range(GPU_NUMS)]   #表示某一时刻GPU上将处理多少请求数，用来记录GPU处理工作负载的压力
        # print(self.combinations)

    def getActionIndexMap(self):    
        for index, action in self.index_action_map.items():
            self.action_index_map[action] = index
        # print(self.action_index_map)

    def showIndexActionMap(self):
        print(self.index_action_map)

    def showActionIndexMap(self):
        print(self.action_index_map)

    # 每获取一个s，输出一个a
    def chooseAction(self, s):
        # s是一个长度为16的Tensor，0维
        # print("state ", s)
        order_index = random.randint(0, len(self.combinations) - 1)
        order = self.combinations[order_index]
        cur_inputstream = collections.defaultdict(int)
        cur_inputstream['resnet50'] = int(s[0]) + int(s[1]) 
        cur_inputstream['vgg19'] = int(s[4]) + int(s[5]) 
        cur_inputstream['densenet201'] = int(s[8]) + int(s[9]) 
        cur_inputstream['mobilenet'] = int(s[12]) + int(s[13]) 
        action = []
        for model_name in order:
            # 获取当前拥有最少负载的GPU上的负载数量
            min_rq = min(self.gpu_state)
            # 获取该负载最少的GPU的编号
            cur_free_gpu_idx = self.gpu_state.index(min_rq)
            action.append((model_name, cur_free_gpu_idx, self.batch))
            self.gpu_state[cur_free_gpu_idx] += cur_inputstream[model_name]
        
        # 按照(('resnet50', 1, 16), ('vgg19', 1, 16), ('densenet201', 1, 8), ('mobilenet', 0, 4))这个顺序重新排列一下action中各个模型的顺序
        action.sort(key = lambda x:self.model_names.index(x[0]))
        action_index = self.action_index_map[tuple(action)]
        return action_index


if __name__ == "__main__":
    # ascend_list = [0, 20, 40, 60, 80, 180]
    ascend_list = [80]
    loop_times = len(ascend_list)
    reward_res = [0 for _ in range(loop_times)]
    dis_precent_res = [[0 for _ in range(4)] for _ in range(loop_times)]
    load_times_res = [[0 for _ in range(4)] for _ in range(loop_times)]

    for i in range(loop_times):
        
        print("loop {}".format(i + 1))
        loop_ascend = ascend_list[i]

        env = gym.make('GPUcluster-1a_p_4m_2g', evaluation_flag = True, input_ascend = loop_ascend)

        scheduler = minWorkloadScheduler(env)

        s = env.reset()
        done = False

        episode_r = 0
        input_stream = env.getOncTimeInputStreamMessage()

        while(not done):
            a = scheduler.chooseAction(s)
            s_, r, done, mes = env.step(a)
            s = s_
            episode_r += r
            if done:
                extra_message, dis_per_load_times = mes[0], mes[1]
                # print(extra_message)
                # print("--------------------------------")
                # print(dis_per_load_times)
        reward_res[i] = episode_r

        dis_precent_res[i][0] = dis_per_load_times['resnet50_dis_percent']
        dis_precent_res[i][1] = dis_per_load_times['vgg19_dis_percent']
        dis_precent_res[i][2] = dis_per_load_times['densenet201_dis_percent']
        dis_precent_res[i][3] = dis_per_load_times['mobilenet_dis_percent']

        load_times_res[i][0] = dis_per_load_times['resnet50_load_times']
        load_times_res[i][1] = dis_per_load_times['vgg19_load_times']
        load_times_res[i][2] = dis_per_load_times['densenet201_load_times']
        load_times_res[i][3] = dis_per_load_times['mobilenet_load_times']

    # cur_time = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # os.mkdir("./{}".format(cur_time))
    
    # with open("./{}/reward_res.csv".format(cur_time), mode="w", encoding="utf-8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["ascend", "reward"])
    #     for idx, row in enumerate(reward_res):
    #         print(ascend_list[idx])
    #         print(row)
    #         writer.writerow([ascend_list[idx], row])

    # with open("./{}/dis_percent_res.csv".format(cur_time), mode="w", encoding="utf-8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["ascend", "resnet50_dis_percent", "vgg19_dis_percent", "densenet201_dis_percent", "mobilenet_dis_percent"])
    #     for idx, row in enumerate(dis_precent_res):
    #         writer.writerow([ascend_list[idx], row[0], row[1], row[2], row[3]])

    # with open("./{}/load_times_res.csv".format(cur_time), mode="w", encoding="utf-8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["ascend", "resnet50_load_times", "vgg19_load_times", "densenet201_load_times", "mobilenet_load_times"])
    #     for idx, row in enumerate(load_times_res):
    #         writer.writerow([ascend_list[idx], row[0], row[1], row[2], row[3]])

    for k, v in extra_message.items():
    # k是五类图像的关键词throughput,buffer,discard,GPUloadcost,GPUbusytime
        if k == 'throughput':
            plt.figure()
            plt.title("Throughput & Input")
            cnt = 1
            for model_name, t_list in v.items():
                plt.subplot(2,2,cnt)
                x = [i for i in range(len(t_list))]
                plt.plot(x, input_stream[model_name], label = "request")
                plt.plot(x, t_list, label = "throughput")
                plt.legend()
                plt.xlabel(model_name)
                plt.ylabel("Number")
                plt.grid(linestyle="-.")
                cnt += 1
            plt.show()
        if k == 'buffer':
            plt.figure()
            plt.title("Buffer Length")
            cnt = 1
            for model_name, t_list in v.items():
                plt.subplot(2,2,cnt)
                x = [i for i in range(len(t_list))]
                plt.plot(x, t_list, label = "Buffer Length")
                plt.legend()
                plt.xlabel(model_name)
                plt.ylabel("Number")
                plt.grid(linestyle="-.")
                cnt += 1
            plt.show()
        if k == 'discard':
            plt.figure()
            plt.title("Discarded Requset Number")
            cnt = 1
            for model_name, t_list in v.items():
                plt.subplot(2,2,cnt)
                x = [i for i in range(len(t_list))]
                plt.plot(x, t_list, label = "Discarded Requset Number")
                plt.legend()
                plt.xlabel(model_name)
                plt.ylabel("Number")
                plt.grid(linestyle="-.")
                cnt += 1
            plt.show()
        if k == 'GPUloadcost':
            plt.figure()
            plt.title("GPU Load Model Cost")
            cnt = 1
            for gpu_idx, t_list in v.items():
                plt.subplot(2,1,cnt)
                x = [i for i in range(len(t_list))]
                plt.plot(x, t_list, label = "GPU Load Model Cost")
                plt.legend()
                plt.xlabel("GPU {}".format(gpu_idx))
                plt.ylabel("Time cost")
                plt.grid(linestyle="-.")
                cnt += 1
            plt.show()
        if k == 'GPUbusytime':
            plt.figure()
            plt.title("GPU Busy Time")
            cnt = 1
            for gpu_idx, t_list in v.items():
                plt.subplot(2,1,cnt)
                x = [i for i in range(len(t_list))]
                plt.plot(x, t_list, label = "GPU Busy Time")
                plt.legend()
                plt.xlabel("GPU {}".format(gpu_idx))
                plt.ylabel("Time(s)")
                plt.grid(linestyle="-.")
                cnt += 1
            plt.show()