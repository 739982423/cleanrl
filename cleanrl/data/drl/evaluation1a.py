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
        action_length = single_env.action_length
        state_length = single_env.observation_length
        # print("action_length", action_length)
        # print("state_length", state_length)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_length, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, action_length), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    env = gym.make('GPUcluster-1a', evaluation = True)
    model = torch.load("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\\drl\\models\\2023-03-08-07-37-47\\a1.pt")

    s = env.reset()
    done = False
    extra_message = None
    input_stream = env.getInputStreamMessage()
    while(not done):
        a, logprob, entropy, c = model.get_action_and_value(torch.Tensor(s))
        print(a) #tensor(8116)
        next_state, reward, done, extra_message = env.step(int(a))
        s = next_state
        if done:
            print(extra_message)
    
    for k, v in extra_message.items():
        # k是五类图像的关键词throughput,buffer,discard,GPUloadcost,GPUbusytime
        if k == 'throughput':
            plt.figure()
            plt.title("Throughput & Input")
            cnt = 1
            for model_name, t_list in v.items():
                plt.subplot(2,3,cnt)
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
                plt.subplot(2,3,cnt)
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
                plt.subplot(2,3,cnt)
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



