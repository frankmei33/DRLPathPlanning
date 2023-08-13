import os, re
import glob
import time
from datetime import datetime
from DoubleGyre_Environment import DoubleGyre_Environment
import matplotlib.pyplot as plt
import sys
import argparse
import pickle

import torch
import numpy as np

import gym
from PPO import PPO
from PPOLSTM import PPOLSTM

################################### Testing ###################################
def test(fix_init, lamb=0.002, rho=0.5, 
          K_epochs = 80, lr_actor = 0.01, lr_critic = 0.02,
          agent = 'PPO', ckpt_id=None):
    print("============================================================================================")

    has_continuous_action_space = True
    max_ep_len = 100
    # env_seed = 1234
    env_name = 'DoubleGyre'

    print("Testing environment name : " + env_name)
    env =  DoubleGyre_Environment(lamb=lamb, rho=rho, fix_init=fix_init) #seed=env_seed, 

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    log_dir = agent + "_logs/" + env_name + "/" + str(fix_init) + '/' + str(lamb) + "_" + str(rho) + '/'
    log_f_name = log_dir + agent + "_" + env_name + "_test_" + str(fix_init) + "_" + str(lamb) + "_" + str(rho) + ".txt"
    log_f_full_name = log_dir + agent + "_" + env_name + "_test_" + str(fix_init) + "_" + str(lamb) + "_" + str(rho) + ("_{}".format(ckpt_id) if ckpt_id is not None else "") + ".pkl"
    print("logging at : " + log_f_full_name)

    ###################### plot ######################
    fig_dir = agent + '_figs/' + env_name + '/' + str(fix_init) + '/' + str(lamb) + "_" + str(rho) + '/'
    if ckpt_id is not None:
        fig_dir += str(ckpt_id) + '/'
    if not os.path.exists(fig_dir):
          os.makedirs(fig_dir)

    print("============================================================================================")

    ################# loading model ################

    # initialize and load a PPO agent 
    # lr_actor = 0.01       # learning rate for actor network 
    # lr_critic = 0.02       # learning rate for critic network
    gamma = 0.99            # discount factor
    # K_epochs = 80               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    action_std = 1.0  
    hidden_layer_dim = 16

    if agent == 'PPO':
        ppo_agent = PPO(state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    elif agent == 'PPOLSTM':
        ppo_agent = PPOLSTM(state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    if ckpt_id is not None: 
        checkpoint_path = agent + "_preTrained/" + env_name + "/{}_{}_{}_{}_{}_{}.pth".format(agent, env_name, fix_init, lamb, rho, ckpt_id)
    else:
        checkpoint_path = agent + "_preTrained/" + env_name + "/{}_{}_{}_{}_{}.pth".format(agent, env_name, fix_init, lamb, rho)
    print("Loading pretrained model: " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    ppo_agent.test_mode()

    print("============================================================================================")

    ################# running agent ################

    log_f = open(log_f_name,"w+")
    log_f_full = open(log_f_full_name,"wb+")

    avg_logdet = 0
    avg_return = 0
    
    for time in np.arange(0,1,0.2):
        state = env.reset(init_t=time)
        current_ep_reward = np.zeros(max_ep_len+1)
        logdets = np.zeros(max_ep_len+1)
        logdets[0] = env.logdet
        returns = np.zeros(max_ep_len+1)
        current_states = [env.s]
        flow = [state[3:5]]

        for t in range(max_ep_len):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action,store_traj=True)
            current_ep_reward[t+1] = reward
            logdets[t+1] = env.logdet
            returns[t+1] = env.prev_dist_to_start
            current_states.append(env.s)
            flow.append(state[3:5])
                
            # break; if the episode is over
            if done:
                break

        avg_logdet += env.logdet 
        avg_return += env.prev_dist_to_start

        summary = {'rewards':current_ep_reward,
                   'logdets':logdets,
                   'returns': returns,
                   'path':env.P.flatten(),
                   'actions':np.array(env.actions),
                   'init_t':time, 
                   'flow':np.vstack(flow), 
                   'state': np.vstack(current_states),
                   'traj': env.traj,
                   }
        pickle.dump(summary, log_f_full)
        log_f_full.flush()

        # save path plot
        plt.figure()
        env.plot_traj() 
        plt.savefig("{}/{}_testing_path_{}_{}_{}_{:.1f}.png".format(fig_dir, env_name,lamb, rho, fix_init, time), bbox_inches='tight')

        # log
        log_f.write('====================================\n')
        log_f.write('Time: {}\n'.format(time))
        log_f.write('Sensor path: {}\n'.format(env.P.flatten()))
        log_f.write('Log determinant of Observability: {}\n'.format(env.logdet))
        log_f.write('Actions: {}\n'.format(np.array2string(np.array(env.actions), max_line_width=1000)))
        log_f.write('Sensor locations: {}\n'.format(np.vstack(current_states)))
        log_f.write('Background Flow: {}\n'.format(np.vstack(flow)))

    log_f.close()
    env.close()

    return avg_logdet/5, avg_return/5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for HYCOM environment and RL model.')
    parser.add_argument('--init', type=int, default=None)
    parser.add_argument('--lamb', type=float, default=0.002)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--agent', type=str, default='PPO')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--ckpt_id', type=int, default=-1)
    args = parser.parse_args()

    if args.repeat:
        n = len([f for f in os.listdir('{}_preTrained/DoubleGyre/'.format(args.agent)) if re.match("{}_DoubleGyre_{}_{}_{}_[0-9]+.pth".format(args.agent, args.init, args.lamb, args.rho), f)])
        print("Number of trials:", n)
        for i in range(n):
            test(args.init, args.lamb, args.rho, agent=args.agent, ckpt_id=i)
    elif args.ckpt_id >= 0:
        test(args.init, args.lamb, args.rho, agent=args.agent, ckpt_id=args.ckpt_id)
    else:
        test(args.init, args.lamb, args.rho, agent=args.agent)
    
    
    
    
    
    
    
