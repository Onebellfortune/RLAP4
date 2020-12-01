import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical, Normal
from shared_adam import SharedAdam
import math
import torch.multiprocessing as mp
import time

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()
GAMMA = 0.95
##############################################################
############ 1. Actor Network, Critic Network 구성 ############
##############################################################

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()

        self.a1 = nn.Linear(OBS_DIM, 200)
        self.mu = nn.Linear(200, ACT_DIM)
        self.std = nn.Linear(200, ACT_DIM)
        self.c1 = nn.Linear(OBS_DIM, 100)
        self.v = nn.Linear(100, 1)
        #self.optimizer=optim.Adam(self.parameters(),lr=0.002)
        self.distribution = torch.distributions.Normal
        self.opt = optim.Adam(self.parameters(), lr=0.002)

    def act(self, x):
        mu, std, value = self.forward(x)
        return mu,std

    def cri(self,x):
        criterion=torch.nn.MSELoss()
        return value

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2*F.tanh(self.mu(a1))
        std = F.softplus(self.std(a1)) + 0.001  # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, std, values

    def loss_func(self, state, action, value):
        self.train()
        mu, sigma, values = self.forward(state)
        advantage = value - values
        Value_loss = advantage.pow(2)

        m = self.distribution(mu, sigma)
        #norm_dist = Normal(mu, sigma)
        #action = norm_dist.sample()
        log_prob = m.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * advantage.detach() + 0.005 * entropy
        Policy_loss = -exp_v
        total_loss = (Policy_loss + Value_loss).mean()
        return total_loss

###########################################################################################
############  2. Local actor 학습(global actor, n_steps 받아와서 학습에 사용합니다.)  ############
###########################################################################################

def Worker(global_actor, n_steps):

    local_actor=ActorCritic()
    opt=global_actor.opt
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')


    #global_opt=optim.Adam(global_actor.parameters(),lr=0.0015)

    total_step=1
    for episode in range(3000):
        state = env.reset()
        buffer_state, buffer_action, buffer_reward = [], [], []
        score = 0.0
        for e in range(200):
            env.render()
            mu, std = local_actor.act(torch.from_numpy(state).float())
            norm_dist = Normal(mu, std)
            action = norm_dist.sample()
            next_state, reward, done, _ = env.step(action)
            score += reward  # normalize
            buffer_action.append(action)
            buffer_state.append(state)
            buffer_reward.append(reward)
            if e==199:
                done=True

            if total_step % 5 == 0 or done:  # update global and assign to local net
                # sync
                if done:
                    Vs = 0.  # terminal
                else:
                    Vs = local_actor.cri(torch.from_numpy(state).float())

                buffer_value_target = []
                for rew in buffer_reward[::-1]:  # reverse buffer r
                    Vs = rew + GAMMA * Vs
                    buffer_value_target.append(Vs)

                buffer_value_target.reverse()
                buffer_state=np.vstack(buffer_state)
                buffer_action=np.array(buffer_action) if buffer_action[0].dtype==np.int64 else np.vstack(buffer_action)
                buffer_value_target=np.array(buffer_value_target)
                buffer_state=buffer_state.astype(np.float32)
                buffer_action=buffer_action.astype(np.float32)
                buffer_value_target=buffer_value_target.astype(np.float32)
                loss = local_actor.loss_func(torch.from_numpy(buffer_state),
                                      torch.from_numpy(buffer_action),
                                      torch.from_numpy(buffer_value_target)[:,None])
                opt.zero_grad()
                loss.backward()
                for lp, gp in zip(local_actor.parameters(), global_actor.parameters()):
                    gp._grad = lp.grad
                opt.step()
                local_actor.load_state_dict(global_actor.state_dict())
                buffer_state, buffer_action, buffer_reward = [], [], []
                if done:  # done and print information
                    score=0.0
                    #record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    break

            state = next_state
            total_step += 1

    env.close()
    print("Training process reached maximum episode.")
    ## 주의 : "InvertedPendulumSwingupBulletEnv-v0"은 continuious action space 입니다.
    ## Asynchronous Advantage Actor-Critic(A3C)를 참고하면 도움이 될 것 입니다.
    
