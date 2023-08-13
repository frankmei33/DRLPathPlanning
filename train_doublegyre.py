import os
import glob
import time
from datetime import datetime
from DoubleGyre_Environment import *
import matplotlib.pyplot as plt
import sys

import torch
import numpy as np
import pickle

import gym
import argparse
# import roboschool

from PPO import PPO
from PPOLSTM import PPOLSTM


def plot_traj(Ps,env,head_width=3,lw=1):
    # plt.figure()
    locs = env._ind_to_grid(Ps.flatten())
    x,y = locs[0], locs[1]
    # plt.figure(figsize=(10,5))
    for i in range(Ps.shape[0]):
        dx = x[(i+1)%len(x)] - x[i]
        dy = y[(i+1)%len(x)] - y[i]
        plt.arrow(x[i],y[i],dx,dy,color='k',lw=lw,head_width=head_width,length_includes_head=True)
    plt.scatter(x,y,color='k',zorder=3,s=4*lw**2)


################################### Training ###################################
def train(fix_init=None, lamb=0.002, rho=0.5, # environment parameters
          K_epochs = 50, update_const = 20, lr_actor = 0.005, lr_critic = 0.01, # RL hyperparameters
          agent = 'PPO', ckpt_id = None, random_seed = 0, env_name = 'DoubleGyre'):
    print("============================================================================================")

    ####### initialize environment ########################
    print("Training environment name : " + env_name)
    print("Lambda : ", lamb)
    print("Rho : ", rho)

    # env_seed = 1234
    if env_name == 'DoubleGyre':
        env = DoubleGyre_Environment(lamb=lamb, rho=rho, fix_init=fix_init) #seed=env_seed, 
    elif env_name == 'DoubleGyreG':
        env = DoubleGyreG_Environment(lamb=lamb, rho=rho) 
        fix_init = 'rand'
    snapshot = env.get_data(0)
    if fix_init is None:
        fix_init = 'rand'

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.shape[0]

    ####### Training hyperparameters ########################

    max_ep_len = env.L                    # max timesteps in one episode
    if fix_init == 'rand':
        max_training_timesteps = int(5e6)   # break training loop if timeteps > max_training_timesteps

        print_freq = max_ep_len * 1000        # print avg reward in the interval (in num timesteps)
        log_freq = max_ep_len * 500           # log avg reward in the interval (in num timesteps)
        save_model_freq = int(5e5)          # save model frequency (in num timesteps)

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len * 500                     # update policy every n timesteps
        hidden_layer_dim = 8

        action_std = 0.4                    # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05         # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.02               # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(5e5)    # action_std decay frequency (in num timesteps)
    else:
        max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

        print_freq = max_ep_len * 200        # print avg reward in the interval (in num timesteps)
        log_freq = max_ep_len * 100           # log avg reward in the interval (in num timesteps)
        save_model_freq = int(1e4)          # save model frequency (in num timesteps)

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len * update_const                     # update policy every n timesteps
        # K_epochs = rlargs['K_epochs'] if 'K_epochs' in rlargs else 80   # update policy for K epochs in one PPO update
        hidden_layer_dim = 8

        action_std = 0.4                    # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05         # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.02               # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(1e4)    # action_std decay frequency (in num timesteps)

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 1            # discount factor

    # lr_actor = 0.01       # learning rate for actor network 
    # lr_critic = 0.02       # learning rate for critic network

    ###################### plot #########################
    fig_dir = agent + "_figs/" + env_name + '/' + str(fix_init) + '/' + str(lamb) + "_" + str(rho) + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = agent + "_logs/" + env_name  + '/' + str(fix_init) + '/' + str(lamb) + "_" + str(rho) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### create new log file for each run
    log_f_name = log_dir + agent + '_' + env_name + "_log_" + str(fix_init) + "_" + str(lamb) + "_" + str(rho) + ".csv"
    log_f_full_name = log_dir + agent + "_" + env_name + "_log_" + str(fix_init) + "_" + str(lamb) + "_" + str(rho) + ("_{}".format(ckpt_id) if ckpt_id is not None else "") + ".pkl"

    print("logging at : " + log_f_full_name)

    ################### checkpointing ###################
    directory = agent + '_preTrained/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    if ckpt_id is not None:
        print("current checkpointing run number for " + env_name + " : ", ckpt_id)
        checkpoint_path = directory + "{}_{}_{}_{}_{}_{}.pth".format(agent, env_name, fix_init, lamb, rho, ckpt_id)
    else:
        checkpoint_path = directory + "{}_{}_{}_{}_{}.pth".format(agent, env_name, fix_init, lamb, rho)
    
    print("save checkpoint path : " + checkpoint_path)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("agent : " + agent)
    print("agent update frequency : " + str(update_timestep) + " timesteps")
    print("agent K epochs : ", K_epochs)
    print("agent epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    if agent == 'PPO':
        ppo_agent = PPO(state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)
    elif agent == 'PPOLSTM':
        ppo_agent = PPOLSTM(state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,avg reward,avg return\n')
    log_f.flush()
    log_f_full = open(log_f_full_name, "wb+")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    running_return = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:

        state = env.reset()
        # print(env.s,state[-2:],env.logdet)
        if agent == 'PPOLSTM':
            ppo_agent.reset_hidden_layer()
        current_ep_reward = 0
        tic = time.time()

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            # print(env.s,action,state[-2:],reward,env.logdet)
            # while done and t < max_ep_len:
            #     action = ppo_agent.select_action(state,redo=True)
            #     state, reward, done, _ = env.step(action)
                # print(env.s,env.P[min(t,51)],state[-2:],action,env.l,env.init_t)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            if agent == 'PPOLSTM':
                ppo_agent.buffer.put_batch_state(done)

            time_step +=1
            # print(action, state, reward)
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                avg_return = running_return / log_running_episodes
                avg_return = round(avg_return, 4)

                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, avg_return))
                log_f.flush()
                summary = {'episode':i_episode,'reward':last_ep_reward,'logdet':current_logdet,
                           'return':current_return,'path':current_path,'actions':current_actions,
                           'init_t':current_time}
                pickle.dump(summary, log_f_full)
                log_f_full.flush()

                log_running_reward = 0
                log_running_episodes = 0
                running_return = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Current Logdet: {}".format(i_episode, time_step, print_avg_reward, current_logdet))

                print_running_reward = 0
                print_running_episodes = 0

                # save path plot
                plt.figure()
                plot_traj(current_path,env)
                plt.imshow(snapshot,origin='lower')
                plt.savefig("{}/running_path_{}_{}_{}.png".format(fig_dir,fix_init,lamb,rho))                

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        # print(i_episode,time.time()-tic)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

        current_return = np.sum((env.s-env.start)**2)
        running_return += current_return
        current_path = env.P.flatten()
        current_actions = np.array(env.actions)
        current_logdet = env.logdet 
        last_ep_reward = current_ep_reward
        current_time = env.init_t

    log_f.close()
    log_f_full.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for HYCOM environment and RL model.')
    parser.add_argument('--init', type=int, default=None)
    parser.add_argument('--lamb', type=float, default=0.002)
    parser.add_argument('--rho', type=float, default=10.0)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--update', type=int, default=20)
    parser.add_argument('--agent', type=str, default='PPO')
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--ckpt_id', type=int, default=-1)
    parser.add_argument('--env_name', type=str, default='DoubleGyre')
    args = parser.parse_args()
    print(args)

    if args.repeat > 0:
        for i in range(args.repeat):
            train(args.init, args.lamb, args.rho, lr_actor=args.lr, lr_critic=args.lr*2, update_const = args.update, agent=args.agent, ckpt_id=i, random_seed=1234+i, env_name=args.env_name)
    elif args.ckpt_id >= 0:
        train(args.init, args.lamb, args.rho, lr_actor=args.lr, lr_critic=args.lr*2, update_const = args.update, agent=args.agent, ckpt_id=args.ckpt_id, random_seed=1234+args.ckpt_id, env_name=args.env_name)
    else:
        train(args.init, args.lamb, args.rho, lr_actor=args.lr, lr_critic=args.lr*2, update_const = args.update, agent=args.agent, env_name=args.env_name,K_epochs = 100)



    
    
    
    
    
    
    
    
