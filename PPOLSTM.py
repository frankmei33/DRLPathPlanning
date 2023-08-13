import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)

def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.batch_states = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.batch_states[:]

    def put_batch_state(self, done):
        if done:
            self.batch_states.append(torch.squeeze(torch.stack(self.states, dim=0)).detach().to(device))
            del self.states[:]

class ActorCritic(nn.Module):
    train = True

    def __init__(self, state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_layer_dim = hidden_layer_dim
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        self.actorLSTM = nn.LSTM(state_dim, self.hidden_layer_dim, batch_first=True)
        self.actorHidden = None
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_layer_dim, action_dim),
                            # nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_layer_dim, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.criticLSTM = nn.LSTM(state_dim, self.hidden_layer_dim, batch_first=True)
        self.criticHidden = None
        self.critic = nn.Sequential(
                        nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                        nn.Tanh(),
                        nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                        nn.Tanh(),
                        nn.Linear(self.hidden_layer_dim, 1)
                    )

        # self.actor.apply(init_weights)
        # self.critic.apply(init_weights)
        orthogonal_init(self.actor)
        orthogonal_init(self.actorLSTM)
        orthogonal_init(self.critic)
        orthogonal_init(self.criticLSTM)
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            output, self.actorHidden = self.actorLSTM(state, self.actorHidden)
            action_mean = self.actor(output)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            output, self.actorHidden = self.actorLSTM(state, self.actorHidden)
            action_probs = self.actor(output)
            dist = Categorical(action_probs)

        if self.train:
            action = dist.sample()
        else:
            action = dist.mean
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        # state needs to be shapes by batch sequences, (batchsize, timelength, inputsize)

        if self.has_continuous_action_space:
            output, _ = self.actorLSTM(state)
            action_mean = self.actor(torch.flatten(output,end_dim=1))
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            output, _ = self.actorLSTM(state)
            action_probs = self.actor(torch.flatten(output,end_dim=1))
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        output, _ = self.criticLSTM(state)
        state_values = self.critic(torch.flatten(output,end_dim=1))
        
        return action_logprobs, state_values, dist_entropy


class PPOLSTM:
    def __init__(self, state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        # adaptive learning rate
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.999)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, redo = False):
        if redo:
            self.buffer.states.pop()
            self.buffer.actions.pop()
            self.buffer.logprobs.pop()

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        batchsizes = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
                batchsizes.insert(0, 1)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            batchsizes[0] += 1
            
        # Normalizing the rewards 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) 

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.batch_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # update lr
        self.lr_scheduler.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def test_mode(self):
        self.policy_old.train = False
        self.policy.train = False
        
    def reset_hidden_layer(self):
        self.policy_old.actorHidden = None
        self.policy_old.criticHidden = None

       


