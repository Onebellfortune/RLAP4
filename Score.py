import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.multiprocessing as mp
from torch.distributions import Normal
import torch
import time
import gym
import argparse
from Agent import Worker, ActorCritic

###########################################################################################
############  3. Evaluate하는 부분(목표 reward에 도달하면 조기종료)  ###################################
###########################################################################################

def Evaluate(global_actor, mode):
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
    score = 0.0
    start_time = time.time()

    for episode in range(3000):
        done = False
        state = env.reset()
        finish = False
        while not done:
            mu, std = global_actor.act(torch.from_numpy(state).float())
            norm_dist = Normal(mu, std)
            action = norm_dist.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

        if episode % 100 == 0 and episode != 0:
            print("Episode: {}, avg score: {:.1f}".format(episode, score/100))
            if mode == "SS" and (score/100) >= 500.:
                finish = True
                print("Solved (1)!!!, Time : {:.2f}".format(time.time() - start_time))
            elif mode == "MS" and (score/100) >= 600.:
                print("Solved (2)!!!, Time : {:.2f}".format(time.time() - start_time))
                finish = True
            elif mode == "MM" and (score/100) >= 700.:
                print("Solved (3)!!!, Time : {:.2f}".format(time.time() - start_time))
                finish = True
            score = 0.0
            time.sleep(1)
        
        if finish:
            break
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate Your Actor Critic Model")
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--multi', type=int, default=1)
    args = parser.parse_args()

    if args.n_steps == 1 and args.multi == 1:
        mode = "SS"
    elif args.n_steps != 1 and args.multi == 1:
        mode = "MS"
    elif args.n_steps !=1 and args.multi != 1:
        mode = "MM"

    global_actor = ActorCritic()
    global_actor.share_memory()

    processes = []
    for rank in range(args.multi + 1):  # + 1 for test process
        if rank == 0: #master process는 evaluate 한거 뿌려주고
            p = mp.Process(target=Evaluate, args=(global_actor, mode))
        else: #나머지 worker들은 training을 하고.
            p = mp.Process(target=Worker, args=(global_actor, args.n_steps,))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()
