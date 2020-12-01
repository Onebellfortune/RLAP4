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
cuda=torch.device('cuda')
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
        return criterion

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

def train_model(actor,critic,actor_optimizer,critic_optimizer,transition,policies):
    state,action,reward,next_state,mask=transition

    criterion=torch.nn.MSELoss()

    value=critic(torch.Tensor(state).cuda()).squeeze(1)
    next_value=critic(torch.Tensor(next_state).cuda()).squeeze(1)
    target=reward+mask*GAMMA*next_value

    critic_loss=criterion(value,target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    categorical=Categorical(policies)
    log_policy=categorical.log_prob(torch.Tensor([action]).cuda())

    advantage=target-value
    actor_loss=-log_policy*advantage.item()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def get_action(policies):
    categorical=Categorical(policies)
    action=categorical.sample()
    action=action.data.numpy()[0]

    return action

###########################################################################################
############  2. Local actor 학습(global actor, n_steps 받아와서 학습에 사용합니다.)  ############
###########################################################################################

def Worker(global_actor, n_steps):

    local_actor=ActorCritic()
    opt=global_actor.opt
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')

    actor=ActorCritic()
    critic=ActorCritic()
    actor_optimizer=optim.Adam(actor.parameters(),lr=0.002)
    critic_optimizer=optim.Adam(critic.parameters(),lr=0.002)


    for episode in range(3000):
        done=False
        score=0
        state = env.reset()
        state=np.reshape(state,[1,OBS_DIM])


        while not done:
            env.render()
            mu, std = local_actor.act(torch.from_numpy(state).float())
            norm_dist = Normal(mu, std)
            action = norm_dist.sample()
            next_state, reward, done, _ = env.step(action)
            next_state=np.reshape(next_state[1,OBS_DIM])
            mask=0 if done else 1
            transition=[state,action,reward,next_state,mask]
            train_model(actor,critic,actor_optimizer,critic_optimizer,transition,policies)
            state=next_state
            score+= reward  # normalize

        running_score=0.99*running_score+0.01*score
        if episode % 5 ==0:
            print('{} episode | running score: {:2f}'.format(episode,running_score))

        if running_score > 500:
            print('over 500')
            break

    env.close()
    print("Training process reached maximum episode.")
    ## 주의 : "InvertedPendulumSwingupBulletEnv-v0"은 continuious action space 입니다.
    ## Asynchronous Advantage Actor-Critic(A3C)를 참고하면 도움이 될 것 입니다.
    
