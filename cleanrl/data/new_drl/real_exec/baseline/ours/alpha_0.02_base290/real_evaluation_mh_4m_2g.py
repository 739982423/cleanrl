import torch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import csv
import os

time_duration = 240
GPU_NUMS = 2
MODEL_NUMS = 6
candidate_batch = [4,8,16]
model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet', 'alexnet', 'inception']   

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, single_env):
        super().__init__()
        exec_action_length = single_env.exec_action_length
        cache_action_length = single_env.cache_action_length
        state_length = single_env.observation_length

        self.base = nn.Sequential(
            layer_init(nn.Linear(state_length, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        self.exec_header = nn.Sequential(
            layer_init(nn.Linear(256, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, exec_action_length), std=0.01),
        )

        self.cache_header = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, cache_action_length), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )


    def get_value(self, s):
        return self.critic(s)

    def get_action_and_value(self, s, action_exec=None, action_cache=None):
        base_out = self.base(s)

        logits_exec = self.exec_header(base_out)
        logits_cache = self.cache_header(base_out)

        probs_exec = Categorical(logits=logits_exec)
        probs_cache = Categorical(logits=logits_cache)

        if action_exec == None and action_cache == None:
            action_exec = probs_exec.sample()
            action_cache = probs_cache.sample()

        logprob_exec = probs_exec.log_prob(action_exec)
        logprob_cache = probs_cache.log_prob(action_cache)

        return action_exec, action_cache, logprob_exec, logprob_cache, self.critic(s)


if __name__ == "__main__":
    env = gym.make('GPUcluster-1a_mh_p_4m_2g_real', evaluation_flag = True, input_ascend =90)
    device = torch.device('cuda')

    action_list = []
    with open("alpha_0.02_base290_action_list.csv", mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            action_list.append(row)

    # model_config = "GPUcluster-1a_mh_p_4m_2g__ppo1a_mh_4m_2g__203__r290.0_v270.0_d270.0_m270.0___input_ascend80__memory_alpha0.001_discard_alpha0.5_new"
    # model = torch.load("/home/hpj/project/cleanrl/cleanrl/data/new_drl/real_gpu_test/model/seed204_base330.pt").to(device)

    s = env.reset()
    done = False
    extra_message = None
    input_stream = env.getOncTimeInputStreamMessage()

    idx = 0
    while(not done):
        cur_action = action_list[idx]
        idx += 1
        next_state, reward, done, extra_message = env.step((int(cur_action[0]), int(cur_action[1])))
        s = next_state
    



    plt.figure()
    x = [i for i in range(240)]
    plt.subplot(2,2,1)
    plt.plot(x, extra_message['resnet50_dis_nums'], label="renset50")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(x, extra_message['vgg19_dis_nums'], label="vgg19")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(x, extra_message['densenet201_dis_nums'], label="densenet201")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(x, extra_message['mobilenet_dis_nums'], label="mobilenet")
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x, extra_message['resnet50_max_throughput'], label="renset50")
    plt.plot(x, input_stream['resnet50'], label="renset50",color='orange')
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(x, extra_message['vgg19_max_throughput'], label="vgg19")
    plt.plot(x, input_stream['vgg19'], label="vgg19",color='orange')
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(x, extra_message['densenet201_max_throughput'], label="densenet201")
    plt.plot(x, input_stream['densenet201'], label="densenet201",color='orange')
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(x, extra_message['mobilenet_max_throughput'], label="mobilenet")
    plt.plot(x, input_stream['mobilenet'], label="mobilenet",color='orange')
    plt.legend()
    plt.show()

    plt.figure()
    x = [i for i in range(240)]
    plt.subplot(2,2,1)
    plt.plot(x, extra_message['resnet50_exec_time'], label="renset50")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(x, extra_message['vgg19_exec_time'], label="vgg19")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(x, extra_message['densenet201_exec_time'], label="densenet201")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(x, extra_message['mobilenet_exec_time'], label="mobilenet")
    plt.legend()
    plt.show()
    
    gpu1_cache_mes = [0 for _ in range(240)]
    gpu2_cache_mes = [0 for _ in range(240)]
    total_cache_mes = extra_message['gpu_available_cache_memory_trace']
    for i in range(len(total_cache_mes)):
        gpu1_cache_mes[i] = total_cache_mes[i][0]
        gpu2_cache_mes[i] = total_cache_mes[i][1]

    plt.figure()
    x = [i for i in range(240)]
    plt.subplot(2,1,1)
    plt.plot(x, gpu1_cache_mes, label="GPU1")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(x, gpu2_cache_mes, label="GPU2")

    plt.legend()
    plt.show()

        
    total_cache_mes = extra_message['total_cache_memory_free']
    print(total_cache_mes)

    if not os.path.exists("./plotdata"):
        os.mkdir("./plotdata")

    with open("./plotdata/dis_nums.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        title = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']
        writer = csv.writer(f)
        writer.writerow(title)
        lines = []
        for i in range(time_duration):
            lines.append([extra_message['resnet50_dis_nums'][i],
                          extra_message['vgg19_dis_nums'][i],
                          extra_message['densenet201_dis_nums'][i],
                          extra_message['mobilenet_dis_nums'][i]])
        for row in lines:
            writer.writerow(row)

    with open("./plotdata/exec_time.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        title = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']
        writer = csv.writer(f)
        writer.writerow(title)
        lines = []
        for i in range(time_duration):
            lines.append([extra_message['resnet50_exec_time'][i],
                          extra_message['vgg19_exec_time'][i],
                          extra_message['densenet201_exec_time'][i],
                          extra_message['mobilenet_exec_time'][i]])
        for row in lines:
            writer.writerow(row)

    with open("./plotdata/memory.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        title = ['GPU0', 'GPU1']
        writer = csv.writer(f)
        lines = []
        for i in range(time_duration):
            lines.append([extra_message['gpu_available_cache_memory_trace'][i][0],
                          extra_message['gpu_available_cache_memory_trace'][i][1]])
        for row in lines:
            writer.writerow(row)