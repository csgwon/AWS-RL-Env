#!/usr/bin/env python

import numpy as np
import math
import random
import sys
from awsenv import *

ACTION_HOLD = 0
ACTION_ADD_ONE = 1
ACTION_ADD_TWO = 2
ACTION_SUBTRACT_ONE = -1
ACTION_SUBTRACT_TWO = -2
actions = [ ACTION_SUBTRACT_TWO, ACTION_SUBTRACT_ONE, ACTION_HOLD, ACTION_ADD_ONE, ACTION_ADD_TWO ]
n_actions = len(actions)

n_num_instance_states = 10

cpu_util_states = [50,70,80,90,101]
requests_per_instance_states = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
network_packets_in_states = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

n_cpu_util_states = len(cpu_util_states)
n_requests_per_instance_states = len(requests_per_instance_states)
n_network_packets_in_states = len(network_packets_in_states)

class AWSQLearner(object):
    def __init__(self, gamma = 0.9, alpha = 0.5, restart=False):
        # gamma factor and step size
        self.gamma = gamma
        self.alpha = alpha

        # simple binning of states (rounding to nearest percent)
        self.stateActionValues = np.zeros((n_num_instance_states, n_cpu_util_states, n_requests_per_instance_states, n_network_packets_in_states, n_actions))
        self.stateActionVisits = np.zeros((n_num_instance_states, n_cpu_util_states, n_requests_per_instance_states, n_network_packets_in_states, n_actions))
        if restart:
            self.stateActionValues = np.load('aws_qlearn_5bin.npy')


    def _get_state_index(self, value, states):
        n_states = len(states)
        for i in range(n_states):
            if value < states[i]:
                return i
        return n_states

    def _get_cpu_util_index(self, cpu_util):
        return self._get_state_index(cpu_util, cpu_util_states)
            
    # epsilon-greedy choice
    def chooseAction(self, state):
        if np.random.binomial(1, 0.1):  # epsilon = 0.1
            return np.random.choice(actions)

        cpu_util_index         = self._get_cpu_util_index(state[1])
        req_per_instance_index = self._get_state_index(state[2], requests_per_instance_states)
        net_packets_in_index   = self._get_state_index(state[3], network_packets_in_states)

        actions_max_value = self.stateActionValues[state[0],
                                                   cpu_util_index,
                                                   req_per_instance_index,
                                                   net_packets_in_index,:].max()
        max_actions = np.array([i for i in range(n_actions) if np.isclose(self.stateActionValues[state[0],
                                                                                                   cpu_util_index,
                                                                                                   req_per_instance_index,
                                                                                                   net_packets_in_index,
                                                                                                   i], actions_max_value, 1e-15) ])
        action_choice = np.random.choice(max_actions)
        return actions[action_choice]

    def _get_action_index(self, currentAction):
        return actions.index(currentAction)
    
    def update(self, currentState, env):
        currentAction = self.chooseAction(currentState)

        obs, reward, _, _ = env.step(currentAction)
        n_instances = obs[0] - 1  # 0-based container
        utilization = obs[1]
        req_per_instance = obs[2]
        net_packets_in = obs[3]
        newState = (max(0, min(n_instances, 9)),
                    int(utilization*100), # convert from decimal to integer for percentage
                    req_per_instance,
                    net_packets_in) 

        current_cpu_util_index         = self._get_cpu_util_index(currentState[1])
        current_req_per_instance_index = self._get_state_index(currentState[2], requests_per_instance_states)
        current_net_packets_in_index   = self._get_state_index(currentState[3], network_packets_in_states)
        new_cpu_util_index         = self._get_cpu_util_index(newState[1])
        new_req_per_instance_index = self._get_state_index(newState[2], requests_per_instance_states)
        new_net_packets_in_index   = self._get_state_index(newState[3], network_packets_in_states)

        self.stateActionVisits[currentState[0],
                               current_cpu_util_index,
                               current_req_per_instance_index,
                               current_net_packets_in_index,
                               self._get_action_index(currentAction)] += 1

        self.stateActionValues[currentState[0],
                               current_cpu_util_index,
                               current_req_per_instance_index,
                               current_net_packets_in_index,
                               self._get_action_index(currentAction)] += self.alpha*(reward +
                                                                                       self.gamma*np.max(self.stateActionValues[newState[0],
                                                                                                                                new_cpu_util_index,
                                                                                                                                new_req_per_instance_index,
                                                                                                                                new_net_packets_in_index,
                                                                                                                                :]) -
                                                                                       self.stateActionValues[currentState[0],
                                                                                                              current_cpu_util_index,
                                                                                                              current_req_per_instance_index,
                                                                                                              current_net_packets_in_index,
                                                                                                              self._get_action_index(currentAction)])
        return newState, reward

    def save(self, name='aws_qlearn.npy'):
        np.save(name, (self.stateActionVisits, self.stateActionValues)) 


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage; ' + sys.argv[0] + ' gamma alpha restart')
        sys.exit(0)

    print('Args: ' + str(sys.argv))
        
    gamma = float(sys.argv[1])
    alpha = float(sys.argv[2])
    restart = False
    if len(sys.argv) > 3:
        if int(sys.argv[3]) > 0:
            restart = True
    
    as_group  = 'AUTOSCALING_GROUP'
    elb_name  = 'ELB_NAME'
    elb_url   = 'ELB_URL'
    env = AWSEnv(as_group=as_group, elb=elb_name, elb_url=elb_url)

    qlearner = AWSQLearner(gamma=gamma, alpha=alpha, restart=restart)
    rewards = []
    states = []
    current_state = (0,0,0,0,0)

    for i in xrange(1000):
        states.append(current_state)
        new_state, reward = qlearner.update(current_state, env)
        rewards.append(reward)
        print('DEBUG: ' + str(current_state) + ";" + str(new_state) + ";" + str(reward))
        current_state = new_state
        qlearner.save()
        np.save('rewards.npy', rewards)
        np.save('states.npy', states)
        if i%12 == 0 and i > 0:
            print('Average reward after ' + str(i) + ' steps: ' + str(float(sum(rewards))/float(i)))
            # checkpoint every step
