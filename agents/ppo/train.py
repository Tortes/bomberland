import os
import torch
import pprint
import argparse
import datetime
import numpy as np
import gym
import asyncio
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from torch.optim.lr_scheduler import LambdaLR
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic
# from tianshou.utils.net.common import Net
from tianshou.utils.net.common import Recurrent
from tianshou.utils import TensorboardLogger

from env import Env
from game_state import GameState

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--server", type=str, default="127.0.0.1:3000")
    parser.add_argument("-a", "--agent", type=str, default="agentA")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--layer-number", type=int, default=128)
    parser.add_argument("--scale-obs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward"
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM"
    )
    return parser.parse_args()

async def main(args=get_args()):
    train_env = Env(args.server, args.agent)
    # Connect client and server
    await asyncio.gather(train_env.server.client.connect(), train_env.server.admin.connect())

    await train_env.server.admin.send_reset()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    action_shape = train_env.action_dim
    state_shape = train_env.state_dim

    # define model
    net_cls = Recurrent
    net = net_cls(
        args.layer_number,
        state_shape,
        action_shape,
        device=args.device
    )
    actor = Actor(net, action_shape, device=args.device, softmax_output=False)
    critic = Critic(net, device=args.device)
    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5
    )

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=train_env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)
    
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_env),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector(policy, train_env, buffer, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo_icm" if args.icm_lr_scale > 0 else "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.task:
            return mean_rewards >= 20
        else:
            return False

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = onpolicy_trainer(
        policy = policy,
        train_collector = train_collector,
        max_epoch = args.epoch,
        step_per_epoch = args.step_per_epoch,
        repeat_per_collect = args.repeat_per_collect,
        episode_per_collect = args.test_num,
        batch_size = args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)


if __name__ == '__main__':
    print(r"============Zelda: Tears Of The Kingdom============")
    print(r"=======================Start=======================")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    asyncio.run(main())
