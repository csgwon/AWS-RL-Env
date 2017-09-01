from awsenv import *
import numpy as np
import tensorflow as tf
import cPickle as pkl
import tensorflow.contrib.slim as slim

def scale_obs(obs):
    obs_2 = obs[2]
    if obs_2 > 0:
        obs_2 = np.log(obs[2])
    obs_3 = obs[3]
    if obs_3 > 0:
        obs_3 = np.log(obs[3])
    return (obs[0]/20.,obs[1],obs_2,obs_3,obs[4])

class Dueling_DQN():
    def __init__(self, D, l, learning_rate):

        self.inputs = tf.placeholder(tf.float32, [None,D, l, 1] , name="input_x")
        self.conv1 = slim.conv2d(\
                    inputs=self.inputs, num_outputs=16, kernel_size=[1, 8], stride=[1, 1], padding='SAME',
                    biases_initializer=None)
        self.maxpool1 = slim.max_pool2d(self.conv1, [1,2])
        self.conv2 = slim.conv2d(\
                    inputs=self.maxpool1, num_outputs=32, kernel_size=[1, 5], stride=[1, 1], padding='SAME',
                    biases_initializer=None)
        self.maxpool2 = slim.max_pool2d(self.conv2, [1,2])

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.maxpool2, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(self.xavier_init([128, O]))
        self.VW = tf.Variable(self.xavier_init([128, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

                # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

                # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, O, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.nextQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)

#Functions that update the parameters of the target network with those of the primary network.
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

as_group  = 'AUTOSCALING_GROUP'
elb_name  = 'ELB_NAME'
elb_url   = 'ELB_URL'
env = AWSEnv(as_group=as_group, elb=elb_name, elb_url=elb_url)

#Env constant
n_num_instance_states = 5
n_cpu_util_states = 101

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

checkpoint_dir = 'training_model_D3Q'
checkpoint_name = os.path.join(checkpoint_dir, 'D3Q')
reload = False
start_i = 0

D = 5
l = 16
O = 5
learning_rate = 1e-4
tau = 0.01  # Rate to update target network toward primary network


# Set up the variable for recording across the episodes.
tf.reset_default_graph()
PQN = Dueling_DQN(D,l, learning_rate) # The primary Q Net
TQN = Dueling_DQN(D,l, learning_rate) # The target Q Net
init = tf.global_variables_initializer()

saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

with tf.Session() as sess:
    if reload:
        saver.restore(sess, checkpoint_name)
    else:
        sess.run(init)
    slist= []
    for i in xrange(l):
        s, r, d, info = env.step(0)
        s = scale_obs(s)
        slist.append(list(s))
        env.step(np.random.randint(0,O)-O//2)
    x = np.asarray(slist).reshape(1, D, l, 1)
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
            a = sess.run(PQN.predict,feed_dict={PQN.inputs:x})
            if np.random.rand(1) < e:
                a[0] = np.random.randint(O)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0]-O//2)
            s1 = scale_obs(s1)
            plot_reward.append(r)  # record reward for plot
            plot_observation.append(s1)
            reward_sum += r
            jList.append(r)
            spop = slist.pop(0)
            slist.append(s1)
            x1 = np.asarray(slist).transpose().reshape(1, D, l, 1)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(PQN.predict,feed_dict={PQN.inputs:x1})
            Q2 = sess.run(TQN.Qout, feed_dict={TQN.inputs: x1})
            doubleQ = Q2[0][Q1[0]]
            #Obtain maxQ' and set our target value for chosen action.
            targetQ = r + y*doubleQ
            #Train our network using target and predicted Q values
            sess.run([PQN.updateModel],feed_dict={PQN.inputs:x,PQN.nextQ:[targetQ,], PQN.actions:a})
            updateTarget(targetOps, sess)
            rAll += r
            x = x1
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'Episode %f Average reward for episode %f.  Total average reward %f.' % (i,
            reward_sum / batch_size, running_reward / batch_size)
        saver.save(sess, checkpoint_name)  # Save the model
        with open('training_model_conv5_D3Q/obs_Q.pkl', 'wb') as fp:
            pkl.dump(plot_observation, fp)
        with open('training_model_conv5_D3Q/reward_Q.pkl', 'wb') as fp:
            pkl.dump(plot_reward, fp)
        with open('training_model_conv5_D3Q/e_Q.pkl', 'wb') as fp:
            pkl.dump((e,i), fp)
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

