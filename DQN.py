from awsenv import *
import numpy as np
import math
import random
import tensorflow as tf
import cPickle as pkl
import shutil
import os 

def scale_obs( obs):
    obs_2 = obs[2]
    if obs_2 > 0:
        obs_2 = np.log(obs[2])
    obs_3 = obs[3]
    if obs_3 > 0:
        obs_3 = np.log(obs[3])
    return (obs[0]/20.,obs[1],obs_2,obs_3,obs[4])

as_group  = 'AUTOSCALING_GROUP'
elb_name  = 'ELB_NAME'
elb_url   = 'ELB_URL'
env = AWSEnv(as_group=as_group, elb=elb_name, elb_url=elb_url)

n_num_instance_states = 5
n_cpu_util_states = 101

# Build the network graph in tensorflow
tf.reset_default_graph()
D = 5
H = 100
O = 5
action_offset = O//2

learning_rate = 1e-4
#This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
inputs1 = tf.placeholder(tf.float32, [None,D] , name="input_x")
with tf.name_scope('q-network'):
    W1 = tf.get_variable("W1", shape=[D, H],
                         initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(inputs1,W1, name='layer1'))
    W2 = tf.get_variable("W2", shape=[H, O],
                         initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1,W2, name='score')
    Qout = tf.nn.sigmoid(score, name='Qout')
    predict = tf.argmax(Qout,1, name='predict')

#From here we define the parts of the network needed for learning a good policy.
with tf.name_scope('loss'):
    nextQ = tf.placeholder(shape=[1,O],dtype=tf.float32,name='nextQ')
    loss = tf.nn.l2_loss(nextQ - score,name='loss')
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='trainer')
    updateModel = trainer.minimize(loss,name='updateModel')

# Set up the variable for recording across the episodes.
init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
reward_sum = 0
num_episodes = 1000
batch_size = 10
plot_reward = [] # for plot bookkeeping
plot_observation = [] # for plot bookkeeping
#create lists to contain total rewards and steps per episode
jList = []
rList = []
running_reward = None
saver = tf.train.Saver()
checkpoint_dir = 'training_model_DQN/'
checkpoint_name = os.path.join(checkpoint_dir, 'DQN')
reload = False
start_i = 0

with tf.Session() as sess:
    sess.run(init)
    s, r, d, info = env.step(0)
    s = scale_obs(s)
    x = np.asarray(s).reshape(1, D)
    reward_sum = 0 
    for i in range(start_i, num_episodes):
        #Reset environment and get first new observation
        rAll = 0
        reward_sum = 0
        d = False
        j = 0
        #The Q-Network
        while j < batch_size:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ,score0 = sess.run([predict,Qout,score],feed_dict={inputs1:x})
            if np.random.rand(1) < e:
                a[0] = np.random.randint(O)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0]-action_offset)
            s1 = scale_obs(s1)
            plot_reward.append(r)  # record reward for plot
            plot_observation.append(s1)
            reward_sum += r
            jList.append(r)
            x1 = np.asarray(s1).reshape(1, D)
            #Obtain the Q' values by feeding the new state through our network
            Q1,score1 = sess.run([Qout,score],feed_dict={inputs1:x1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(score1)
            targetQ = score0
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            loss_,_ = sess.run([loss,updateModel],feed_dict={inputs1:x,nextQ:targetQ})
            rAll += r
            x = x1
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'Episode %f Average reward for episode %f.  Total average reward %f.' % (i,
            reward_sum / batch_size, running_reward / batch_size)
        saver.save(sess, checkpoint_name)  # Save the model
        with open(os.path.join(checkpoint_dir, 'obs.pkl'), 'wb') as fp: pkl.dump(plot_observation, fp)
        with open(os.path.join(checkpoint_dir, 'reward.pkl'), 'wb') as fp: pkl.dump(plot_reward, fp)
        with open(os.path.join(checkpoint_dir, 'e_Q.pkl'), 'wb') as fp: pkl.dump((e,i), fp)
        #Reduce chance of random action as we train the model.
        e = 1./((i/50) + 10)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

##############################################
# import matplotlib.pyplot as plt
# import pandas as pd

# df = pd.DataFrame(jList)
# plt.figure();plt.subplot(121);plt.scatter(xrange(len(df.rolling(12).mean())),df.rolling(12).mean())
# plt.subplot(122);plt.plot(df.rolling(12).mean())


##############################################
# import matplotlib.pyplot as plt
# import pandas as pd

# df = pd.DataFrame(jList)
# plt.figure();plt.subplot(121);plt.scatter(xrange(len(df.rolling(12).mean())),df.rolling(12).mean())
# plt.subplot(122);plt.plot(df.rolling(12).mean())

