# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import sys
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="GPUcluster-4a",
        help="the id of the environment")
    #total_timesteps
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    #num_envs
    parser.add_argument("--num-envs", type=int, default=5,
        help="the number of parallel game environments")
    #num_steps
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    #anneal_lr
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    #gamma
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    #gae_lambda
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    #num_minibatches
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    #update_epochs
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, single_env):
        super().__init__()
        action_length = single_env.action_length
        state_length = single_env.observation_length
        self.actor1 = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_length), std=0.01),
        )
        self.actor2 = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_length), std=0.01),
        )
        self.actor3 = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_length), std=0.01),
        )
        self.actor4 = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_length), std=0.01),
        )

        # print("action vector shape: ", envs.single_action_space.shape)
        # print("state vector length: ", state_length)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_length, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits1 = self.actor1(x)
        logits2 = self.actor2(x)
        logits3 = self.actor3(x)
        logits4 = self.actor4(x)
    
        probs1 = Categorical(logits=logits1)
        probs2 = Categorical(logits=logits2)
        probs3 = Categorical(logits=logits3)
        probs4 = Categorical(logits=logits4)

        if action is None:
            action1 = probs1.sample()
            action2 = probs2.sample()
            action3 = probs3.sample()
            action4 = probs4.sample()

            action1_log_prob = probs1.log_prob(action1)
            action2_log_prob = probs2.log_prob(action2)
            action3_log_prob = probs3.log_prob(action3)
            action4_log_prob = probs4.log_prob(action4)

            action1_entropy = probs1.entropy()
            action2_entropy = probs2.entropy()
            action3_entropy = probs3.entropy()
            action4_entropy = probs4.entropy()

            # action, probs.log_prob(action)与probs.entropy()的形状均相同，在没有修改该函数的初始状态下
            # 它们的输出和形状如下：
            # tensor([0, 0, 1, 1]) torch.Size([4])
            # tensor([-0.6092, -0.6273, -0.6773, -0.6124]) torch.Size([4])
            # tensor([0.6893, 0.6908, 0.6930, 0.6896]) torch.Size([4])

            # print("act 1", action1)
            # print("act 2", action2)
            # print("act 3", action3)
            # print("act 4", action4)
            action = []
            log_probs = []
            probs_entropy = []
            for j in range(len(x)):
                # b_action = []
                # b_logprobs = []
                # b_probs_entropy = []

                # b_action.append(action1[j])
                # b_action.append(action2[j])
                # b_action.append(action3[j])
                # b_action.append(action4[j])

                # b_logprobs.append(action1_log_prob[j])
                # b_logprobs.append(action2_log_prob[j])
                # b_logprobs.append(action3_log_prob[j])
                # b_logprobs.append(action4_log_prob[j])

                # b_probs_entropy.append(action1_entropy[j])
                # b_probs_entropy.append(action2_entropy[j])
                # b_probs_entropy.append(action3_entropy[j])
                # b_probs_entropy.append(action4_entropy[j])

                action.append([action1[j], action2[j], action3[j], action4[j]])
                log_probs.append([action1_log_prob[j], action2_log_prob[j], action3_log_prob[j], action4_log_prob[j]])
                probs_entropy.append([action1_entropy[j], action2_entropy[j], action3_entropy[j], action4_entropy[j]])

            # print("entropy:", torch.Tensor(probs_entropy).type(), torch.Tensor(probs_entropy).shape)
            return torch.Tensor(action), torch.Tensor(log_probs), torch.Tensor(probs_entropy), self.critic(x)

        else:
            # 本函数中，if的两个分支唯一的不同点是：action是否是外部传来的
            # 如果是外部传来的，则进入本分支。
            # 因为外部传来的action，一定是一个带有batch的action，即形状为[b, 4]，b是传来多少组action
            # 每组action，有四个动作，用四个数字表示，索引为0处的数字，是由actor1生产的，索引为3的是由actor4生产的
            # 如果不是外部传来的，则说明action=None，需要在本函数内进行sample获得action
            # sample之后，就会得到每个batch下的四个数字，有多少batch，得到的就是形状为多少batch的action，即[b, 4]
            # 因此，这两个分支本质上的差别只有一处，action是否需要在本函数内sample，如果需要，则sample获得[b, 4]
            # 形状的action，如果不需要，则传入的action就是个[b, 4]形状的action

            # 获取到action之后要做的事情是统一的：求logprob和entropy
            # 因为action是[b, 4]的，所以最终的logprob也应该是[b, 4]的，代表了每个batch中四个动作在对应actor的输出
            # 概率分布中 获取到当前动作 的概率 的log值。每个batch中的logprob是[1,4]的，表示这个batch的四个动作
            # 数字，在对应probs = Categorical(logits=logits)这个分布中sample到的概率的log值，一共有b个batch，所以最终
            # 会获得[b, 4]的logprob Tensor

            log_probs = []
            probs_entropy = []
            # 现在程序卡在417行，现在需要把417行输入action情况的返回值搞明白，此时输入是160个(s,a,r)组合
            # 的list，其中每个s是1行16列的向量，每个a是1行4列的向量
            # 现在需要把每个组合的action[a1,a2,a3,a4]中的a1通入probs1计算logprob，a2a3a4同理，
            # 得到一个160行4列的logprob输出矩阵，以及一个160行4列(四个prob concat起来)?的entropy输出矩阵
            # print("logits1", logits1.type(), logits1.shape)
            # print("probs1", probs1)
            action1 = []
            action2 = []
            action3 = []
            action4 = []
            for i in range(len(x)):
                cur_loop_action = action[i]
                action1.append(cur_loop_action[0])
                action2.append(cur_loop_action[1])
                action3.append(cur_loop_action[2])
                action4.append(cur_loop_action[3])

            # 转换为Tensor
            action1 = torch.Tensor(action1)
            action2 = torch.Tensor(action2)
            action3 = torch.Tensor(action3)
            action4 = torch.Tensor(action4)

            action1_log_prob = probs1.log_prob(action1)
            action2_log_prob = probs2.log_prob(action2)
            action3_log_prob = probs3.log_prob(action3)
            action4_log_prob = probs4.log_prob(action4)
    
            action1_entropy = probs1.entropy()
            action2_entropy = probs2.entropy()
            action3_entropy = probs3.entropy()
            action4_entropy = probs4.entropy()

            # print("acttt1", action1.type(), action1.shape)
            # print("logprob act111", action1_log_prob.type(), action1_log_prob.shape)
            for j in range(len(x)):
                log_probs.append([action1_log_prob[j], action2_log_prob[j],
                                  action3_log_prob[j], action4_log_prob[j]])
                probs_entropy.append([action1_entropy[j], action2_entropy[j], 
                                      action3_entropy[j], action4_entropy[j]])
            log_probs = torch.Tensor(log_probs)
            probs_entropy = torch.Tensor(probs_entropy)
        
            # print("log_probs 111", log_probs.type(), log_probs.shape)
            # print("probs_entropy 111", probs_entropy.type(), probs_entropy.shape)

            return _, torch.Tensor(log_probs), torch.Tensor(probs_entropy), self.critic(x)



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    #创建tensorboard存储对象
    writer = SummaryWriter(f"runs/{run_name}")
    #tensorboard中添加文本字符串，这里是将所有超参数存成了个表格
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 使用GPU训练时，需要固定随机源以保证结果可复现
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    single_env = gym.make(args.env_id)
    agent = Agent(envs, single_env).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (single_env.observation_length,)).to(device)
    # print("obs", obs.type(), obs.shape)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # print("origin actions ", actions.type(), actions.shape)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # print("rewards ", rewards.type(), rewards.shape)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    # print("next obs", next_obs)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    few_episode_return = 0
    few_episode_end_time = 0
    few_episode_nums = 100

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # print("action:", action)
                # print("logprob:", logprob)
                # print("value:", value)
                # action: tensor([0, 0, 0, 1])
                # logprob: tensor([-0.6934, -0.6928, -0.6933, -0.6929])
                # value: tensor([[ 0.0906],
                #                [-0.2004],
                #                [ 0.0810],
                #                [ 0.1793]])
            # values是4行1列，所以需要flatten，具体有多少行由子环境数量决定，但都是一列，flatten后与action和logprob的形状就一致了，均为1行x列
            # 在这里，将玩一步游戏返回的各个参数存入了预置好的变量中，并用step索引，step最大值是args.num_steps，在这里就是玩args.num_steps步之后，才更新一次网络参数
            # 玩args.num_steps步的过程中，收集所有的参数轨迹，组成长度为args.num_steps的list（182-186行定义的参数），以供后续更新参数使用
            values[step] = value.flatten()
            # print("act", action, action.type(), action.shape)
            # print("logprob", logprob, logprob.type(), logprob.shape)
            actions[step] = action
            logprobs[step] = logprob
            # print("values", values.type(), values.shape)
            # print("actions", actions.type(), actions.shape)
            # print("logprobs", logprobs.type(), logprobs.shape)
            # print("rewards", rewards.type(), rewards.shape)
            # TRY NOT TO MODIFY: execute the game and log data.
            # 上面是根据当前的状态让网络做决策，得到a和v，并存入容器
            # 接下来还要根据a获取下一个时刻的s以及r等变量
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # 获取到r后也存入容器，供后续更新使用
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # 将变量转换为tensor并放置到GPU上
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # 如果游戏结束了（done=true），则记录输出episodie的return，并用tensorboard记录当前episodie的return和走过的长度（这部分有待验证）
            for item in info:
                # print(item)
                if "episode" in item.keys():
                    few_episode_return += item['episode']['r']
                    few_episode_end_time += item['time']
                    # print(f"global_step={global_step}, episodic_return={item['episode']['r']}, end_time={item['time']}")
                    if global_step % few_episode_nums == 0:
                        print(f"global_step={global_step}, few_episodic_return={few_episode_return / few_episode_nums}")
                        few_episode_return = 0
                        few_episode_end_time = 0
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # args.num_steps的循环结束了，代表游戏已经玩了args.num_steps步了，先不继续玩了，需要更新网络参数了
        # bootstrap value if not done
        with torch.no_grad():
            # 为了更新参数，还需要获取下一时刻的v（critic的输出）
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # 用于更新actor的一部分参数是advantage，首先建立一个advantage的容器
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            #
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # 这里delta就是优势函数的各步估计，t=127时上面的if走分支1，得到nextvalues=next_value，即t=128时刻的V，则此时的优势函数为1步估计，A_delta=r127 + gamma*V128 - V127(td(1) target)
                # 同理，t为其他值时，nextvalues=values[t + 1]，即128个时刻内，每个时刻得到的V（此时用的V是循环内t的下一个时刻t+1的V），此时的优势函数就是多步估计，
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        # print("advantage:", advantages.shape)
        # print("returns:", returns.shape)
        # print("1")
        # sys.exit()

        # flatten the batch
        # print("obs1111", obs, obs.type(), obs.shape)
        b_obs = obs.reshape((-1,) + (single_env.observation_length,))
        b_logprobs = logprobs.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # 这里reshape之后，所有容器中的数据都是以一维形式排列，[1,2,3,4]，obs和action因为在一个时刻可能是一个向量，所以它们对应的容器是二维的，将每个时刻的obs和action看作一个数字的话，那么每个容器内
        # 的数据都是[t1_data, t2_data, ... tn_data]这样排列的
        # print("b_obs2222", b_obs, b_obs.type(), b_obs.shape)
        # print("b_logprobs", b_logprobs.type(), b_logprobs.shape)
        # print("b_actions", b_actions)
        # print("b_advantages", b_advantages)
        # print("b_returns", b_returns)
        # print("b_values", b_values)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        # b_inds=[1,2,3,...,512]
        clipfracs = []
        for epoch in range(args.update_epochs):
            # 对b_inds进行随机打乱
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                # start从0开始，到batchsize=512结束，步长为minibatch_size，128
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print("mb_inds", mb_inds)
                # print("b_obs[mb_inds]", b_obs[mb_inds].size(),len(b_obs[mb_inds]))
                # print("b_actions.long()[mb_inds]",b_actions.long()[mb_inds].size())
                # print("b_logprobs[mb_inds]", b_logprobs[mb_inds].shape)

                # 这里的mb_inds是索引的序列，索引被shuffle过了，因此该序列内的索引是随机排列的[0,511]之间的数
                # b_obs[mb_inds]是个[128,4]的输入tensor，输入给critic，将输出128个value，组成[128,1]的二维输出张量
                # b_actions.long()[mb_inds]是128个动作组成的tensor(只有1维,128个元素的list)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                # print("newlogprob", newlogprob, newlogprob.size())
                # print("entropy", entropy, entropy.size())
                # print("newvalue", newvalue, newvalue.size())
                # newlogprob 和 entropy 的 size = [128] 是个长度为128的一维张量
                # newvalue 的 size = [128,1] 是critic网络对batch=128的输入产生的输出

                # 计算旧概率分布的log值与新概率分布的log值之差
                logratio = newlogprob - b_logprobs[mb_inds]

                ratio = logratio.exp()
                # print("old b_logprobs", b_logprobs[mb_inds].shape)
                # print("new b_logprobs", newlogprob.shape)
                # print("logratio", logratio.shape)
                # print("ratio", ratio.shape)
                

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # print("b_advantages", b_advantages.shape)
                # print("mb_advantages", mb_advantages.shape)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ratio1 = []
                ratio2 = []
                ratio3 = []
                ratio4 = []
                for i in range(len(ratio)):
                    ratio1.append(ratio[i][0])
                    ratio2.append(ratio[i][1])
                    ratio3.append(ratio[i][2])
                    ratio4.append(ratio[i][3])

                ratio1 = torch.Tensor(ratio1)
                ratio2 = torch.Tensor(ratio2)
                ratio3 = torch.Tensor(ratio3)
                ratio4 = torch.Tensor(ratio4)

                # Policy loss
                actor1_pg_loss1 = -mb_advantages * ratio1
                actor1_pg_loss2 = -mb_advantages * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef)
                actor1_pg_loss = torch.max(actor1_pg_loss1, actor1_pg_loss2).mean()

                actor2_pg_loss1 = -mb_advantages * ratio2
                actor2_pg_loss2 = -mb_advantages * torch.clamp(ratio2, 1 - args.clip_coef, 1 + args.clip_coef)
                actor2_pg_loss = torch.max(actor2_pg_loss1, actor2_pg_loss2).mean()


                actor3_pg_loss1 = -mb_advantages * ratio3
                actor3_pg_loss2 = -mb_advantages * torch.clamp(ratio3, 1 - args.clip_coef, 1 + args.clip_coef)
                actor3_pg_loss = torch.max(actor3_pg_loss1, actor3_pg_loss2).mean()


                actor4_pg_loss1 = -mb_advantages * ratio4
                actor4_pg_loss2 = -mb_advantages * torch.clamp(ratio4, 1 - args.clip_coef, 1 + args.clip_coef)
                actor4_pg_loss = torch.max(actor4_pg_loss1, actor4_pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = actor1_pg_loss + actor2_pg_loss + actor3_pg_loss + actor4_pg_loss -\
                       args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", actor1_pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
