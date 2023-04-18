import torch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym

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
    env = gym.make('GPUcluster-1a_mh_p_4m_2g', evaluation_flag = True, input_ascend = 0)
    device = torch.device('cuda')

    model_config = "GPUcluster-1a_mh_p_4m_2g__ppo1a_mh_4m_2g__204__r210.0_v190.0_d190.0_m190.0___input_ascend0__memory_alpha0.001_discard_alpha0.5_gamma0.95_new"

    model = torch.load("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\\new_drl\\runs_2g_tweet_input_4.3\\{}\\a1.pt".format(model_config)).to(device)
    s = env.reset()
    done = False
    extra_message = None
    input_stream = env.getOncTimeInputStreamMessage()
    while(not done):
        action_exec, action_cache, logprob_exec, logprob_cache, c = model.get_action_and_value(torch.Tensor(s).to(device))
        print(action_exec, action_cache)
        next_state, reward, done, extra_message = env.step((int(action_exec), int(action_cache)))
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