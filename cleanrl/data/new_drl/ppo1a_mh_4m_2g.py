# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import datetime as dt
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
    parser.add_argument("--env-id", type=str, default="GPUcluster-1a_mh_p_4m_2g",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    #num_envs
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    #num_steps
    parser.add_argument("--num-steps", type=int, default=256,
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
    # --- 新增 ----
    parser.add_argument("--discard-alpha", type=float, default=0.1,
        help="请求发生丢弃时计算reward的系数")
    parser.add_argument("--load-alpha", type=float, default=0.5,
        help="模型发生加载时计算reward的系数")
    parser.add_argument("--memory-alpha", type=float, default=0.005,
        help="模型卸载腾出的内存带来的reward系数")
    parser.add_argument("--input-ascend", type=int, default=0,
        help="训练时为每个模型每个时刻的请求到达率增加input_ascend的大小")
    # ---- END ----
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, discard_cost_alpha = args.discard_alpha, memory_reward_alpha = args.memory_alpha, input_ascend = args.input_ascend)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


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

    # def get_action(self, s):
    #     base_out = self.base(s)
    #     prob_exec = self.exec_header(base_out)
    #     prob_cache = self.cache_header(base_out)

    #     dist_exec = Categorical(prob_exec)
    #     dist_cache = Categorical(prob_cache)

    #     exec_action = dist_exec.sample()
    #     cache_action = dist_cache.sample()

    #     logprob_exec = dist_exec.log_prob(exec_action)
    #     logprob_cache = dist_cache.log_prob(cache_action)

    #     return exec_action, cache_action, logprob_exec, logprob_cache


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
    args = parse_args()

    single_env = gym.make(args.env_id, input_ascend = args.input_ascend)
    flag, input_message = single_env.getInputStreamMessage()
    res_file_name = ""
    if flag == "poisson":
        # flag为泊松的时候，input_message为环境中的self.lambda_base，是个字典，k是模型名，v是一个base值
        # self.lambda_base['resnet50'] = 100 + self.input_ascend
        # self.lambda_base['vgg19'] = 90 + self.input_ascend
        for k, v in input_message.items():
            res_file_name += k[0] + str(v) + "_"

    elif flag == "tweet":
        # flag为推特的时候，input_message为环境中的self.input_streams，是个字典，k是模型名，v是该模型的输入序列[100,101,...87,92]
        # self.input_streams['resnet50'] = self.tweet_input['resnet50']
        # self.input_streams['vgg19'] = self.tweet_input['vgg19']
        input_streams = input_message
        for k, v in input_streams.items():
            # 因为v是一个序列，所以这里取序列的第一个值作为显示的base值
            res_file_name += k[0] + str(v[0]) + "_"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{res_file_name}__input_ascend{args.input_ascend}__memory_alpha{args.memory_alpha}_discard_alpha{args.discard_alpha}_gamma{args.gamma}"

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
    store_file = f"runs_2g_tweet_input_4.3/{run_name}_new"
    writer = SummaryWriter(store_file)
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
    # device = torch.device("cuda")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    agent = Agent(single_env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 存储当前使用环境的各个模型输入流信息
    flag, input_message = single_env.getInputStreamMessage()
    if flag == "poisson":
        writer.add_text(
            "Env Input",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in input_message.items()])),
        )
    elif flag == "tweet":
        input_streams = input_message
        for k, v in input_streams.items():
            model_name = k
            input_stream = v
            for i in range(len(input_stream)):
                writer.add_scalar("inputs/{}".format(model_name), input_stream[i], i)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (single_env.observation_length,)).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions_exec = torch.zeros((args.num_steps, args.num_envs)).to(device)
    actions_cache = torch.zeros((args.num_steps, args.num_envs)).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_exec = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_cache = torch.zeros((args.num_steps, args.num_envs)).to(device)

    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

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
                action_exec, action_cache, logprob_exec, logprob_cache, value = agent.get_action_and_value(next_obs)
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
            actions_exec[step] = action_exec
            actions_cache[step] = action_cache
            logprobs_exec[step] = logprob_exec
            logprobs_cache[step] = logprob_cache

            # actions[step] = action
            # print("action", action)
            # logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # 上面是根据当前的状态让网络做决策，得到a和v，并存入容器
            # 接下来还要根据a获取下一个时刻的s以及r等变量
            # print("action.cpu().numpy()", action.cpu().numpy())
            # next_obs, reward, done, info = envs.step(exec_action = action_exec.cpu().numpy(), 
            #                                          cache_action = action_cache.cpu().numpy())

            # print("action_exec", action_exec.shape, action_exec)
            # print("action cache", action_cache.shape, action_cache)

            pair_action = []
            for i in range(args.num_envs):
                pair_action.append((action_exec[i].cpu().numpy(), action_cache[i].cpu().numpy()))
            # print("pair_action", pair_action)
            next_obs, reward, done, info = envs.step(pair_action)
            
            # 获取到r后也存入容器，供后续更新使用
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # 将变量转换为tensor并放置到GPU上
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # 如果游戏结束了（done=true），则记录输出episodie的return，并用tensorboard记录当前episodie的return和走过的长度（这部分有待验证）
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

                    writer.add_scalar("memory/gpu1_cache_memory_free", item["gpu1_cache_memory_free"], global_step)
                    writer.add_scalar("memory/gpu2_cache_memory_free", item["gpu2_cache_memory_free"], global_step)
                    writer.add_scalar("memory/total_cache_memory_free", item["total_cache_memory_free"], global_step)

                    writer.add_scalar("exec_time/total_exec_time", item["total_exec_time"], global_step)

                    writer.add_scalar("dis_nums/resnet50_dis_nums", item["resnet50_dis_nums"], global_step)
                    writer.add_scalar("dis_percent/resnet50_dis_percent", item["resnet50_dis_percent"], global_step)
                    writer.add_scalar("load_times/resnet50_load_times", item["resnet50_load_times"], global_step)
                    writer.add_scalar("unload_times/resnet50_unload_times", item["resnet50_unload_times"], global_step)
                    writer.add_scalar("exec_time/resnet50_exec_time", item["resnet50_exec_time"], global_step)

                    writer.add_scalar("dis_nums/vgg19_dis_nums", item["vgg19_dis_nums"], global_step)
                    writer.add_scalar("dis_percent/vgg19_dis_percent", item["vgg19_dis_percent"], global_step)
                    writer.add_scalar("load_times/vgg19_load_times", item["vgg19_load_times"], global_step)
                    writer.add_scalar("unload_times/vgg19_unload_times", item["vgg19_unload_times"], global_step)
                    writer.add_scalar("exec_time/vgg19_exec_time", item["vgg19_exec_time"], global_step)

                    writer.add_scalar("dis_nums/densenet201_dis_nums", item["densenet201_dis_nums"], global_step)
                    writer.add_scalar("dis_percent/densenet201_dis_percent", item["densenet201_dis_percent"], global_step)
                    writer.add_scalar("load_times/densenet201_load_times", item["densenet201_load_times"], global_step)
                    writer.add_scalar("unload_times/densenet201_unload_times", item["densenet201_unload_times"], global_step)
                    writer.add_scalar("exec_time/densenet201_exec_time", item["densenet201_exec_time"], global_step)

                    writer.add_scalar("dis_nums/mobilenet_dis_nums", item["mobilenet_dis_nums"], global_step)
                    writer.add_scalar("dis_percent/mobilenet_dis_percent", item["mobilenet_dis_percent"], global_step)
                    writer.add_scalar("load_times/mobilenet_load_times", item["mobilenet_load_times"], global_step)
                    writer.add_scalar("unload_times/mobilenet_unload_times", item["mobilenet_unload_times"], global_step)
                    writer.add_scalar("exec_time/mobilenet_exec_time", item["mobilenet_exec_time"], global_step)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_logprobs_exec = logprobs_exec.reshape(-1)
        b_logprobs_cache = logprobs_cache.reshape(-1)

        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_actions_exec = actions_exec.reshape((-1,))
        b_actions_cache = actions_cache.reshape((-1,))

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # 这里reshape之后，所有容器中的数据都是以一维形式排列，[1,2,3,4]，obs和action因为在一个时刻可能是一个向量，所以它们对应的容器是二维的，将每个时刻的obs和action看作一个数字的话，那么每个容器内
        # 的数据都是[t1_data, t2_data, ... tn_data]这样排列的
        # print("b_obs2222", b_obs, b_obs.type(), b_obs.shape)
        # print("b_logprobs", b_logprobs)
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
                # print("mb_inds", mb_inds, mb_inds.shape)
                # print("b_obs[mb_inds]", b_obs[mb_inds].size(),len(b_obs[mb_inds]))
                # print("b_actions.long()[mb_inds]",b_actions.long()[mb_inds].size())
                # 这里的mb_inds是索引的序列，索引被shuffle过了，因此该序列内的索引是随机排列的[0,511]之间的数
                # b_obs[mb_inds]是个[128,4]的输入tensor，输入给critic，将输出128个value，组成[128,1]的二维输出张量
                # b_actions.long()[mb_inds]是128个动作组成的tensor(只有1维,128个元素的list)
                action_exec, action_cache, new_logprob_exec, new_logprob_cache, newvalue  = agent.get_action_and_value(b_obs[mb_inds], 
                                                                                                                        action_exec = b_actions_exec.long()[mb_inds], 
                                                                                                                        action_cache = b_actions_cache.long()[mb_inds])

                # print("newlogprob", newlogprob.size())
                # print("entropy", entropy.size())
                # print("newvalue", newvalue.size())
                # newlogprob 和 entropy 的 size = [128] 是个长度为128的一维张量
                # newvalue 的 size = [128,1] 是critic网络对batch=128的输入产生的输出

                # 计算旧概率分布的log值与新概率分布的log值之差
                # logratio = newlogprob - b_logprobs[mb_inds]
                # ratio = logratio.exp()

                exec_logratio = new_logprob_exec - b_logprobs_exec[mb_inds]
                exec_ratio = exec_logratio.exp()

                cache_logratio = new_logprob_cache - b_logprobs_cache[mb_inds]
                cache_ratio = cache_logratio.exp()


                # with torch.no_grad():
                #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
                #     old_approx_kl = (-logratio).mean()
                #     approx_kl = ((ratio - 1) - logratio).mean()
                #     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # action exec
                exec_pg_loss1 = -mb_advantages * exec_ratio
                exec_pg_loss2 = -mb_advantages * torch.clamp(exec_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                exec_pg_loss = torch.max(exec_pg_loss1, exec_pg_loss2).mean()

                # action cache
                cache_pg_loss1 = -mb_advantages * cache_ratio
                cache_pg_loss2 = -mb_advantages * torch.clamp(cache_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                cache_pg_loss = torch.max(cache_pg_loss1, cache_pg_loss2).mean()

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

                # entropy_loss = entropy.mean()
                loss = exec_pg_loss + cache_pg_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", exec_pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # current_time = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # os.mkdir(r"./runs/{}".format(run_name))
    torch.save(agent, r"./{}/a1.pt".format(store_file))