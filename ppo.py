"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

"""

from inc import *

run_time=round(time.time())
LOAD = True ## True: no train and use previous pretrained model|| False: Train the model
EP_MAX = 10000
EP_LEN = 800
GAMMA = 0.9
A_LR = 0.0002
C_LR = 0.0002
BATCH = 64
A_UPDATE_STEPS = 8
C_UPDATE_STEPS = 8
#DISCRETE_ACTION = False
List_goald = [(400,600,400,800), (50,150,400,600)]

start = [random.randint(50,600),random.randint(20,50)]
goald = [random.randint(400,600),random.randint(400,800)]
env = CarDCENV(map_bin=mask, goald=goald,start_point=start)
#env = CarEnv(discrete_action=DISCRETE_ACTION)
S_DIM, A_DIM = env.state_dim_, env.action_dim
ACTION_BOUND = env.action_bound
METHOD = dict(name='clip', epsilon=0.2)                # Clipped surrogate objective, find this is better
															# epsilon=0.2 is in the paper
np.random.seed(7)
class PPO(object):
    def __init__(self,sess,LOAD):
        self.sess = sess #tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # CRITIC #######################################
        with tf.variable_scope('critic'):
            l0 =  tf.layers.batch_normalization(self.tfs)
            l1 = tf.layers.dense(l0, 200, tf.nn.relu, name='layer1-critic')
            l10 =  tf.layers.batch_normalization(l1)
            l2 = tf.layers.dense(l10, 100, tf.nn.relu, name='layer2-critic')
            l20 =  tf.layers.batch_normalization(l2)
            self.v = tf.layers.dense(l20, 1, name = 'V_layer')

            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs) # insted SGD

        # ACTOR ########################################
        # Current policy
        pi, pi_params = self._build_anet('pi', trainable=True)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action

        # Hold policy
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('update_oldpi'): # Swap the layer weights of hold_pi by pi
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # PPO implementation, Loss function ############
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate_pp'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                # ratio = probability ratio
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        # Implementation the Training method ############## 
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        
        # Save the model into a tensorboard file
        tf.summary.FileWriter("logs/", self.sess.graph)
        
        # Load the pre training model or just initialize a new training
        saver = tf.train.Saver()
        path = './models'
        if LOAD:
            saver.restore(sess, tf.train.latest_checkpoint(path))
        else:
            sess.run(tf.global_variables_initializer())


    def update(self, s, a, r):
        # Update old policy
        self.sess.run(self.update_oldpi_op)
        
        # Calculate advantage
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable): # Build the current & hold structure for the policies 
        with tf.variable_scope(name):
            l0 =  tf.layers.batch_normalization(self.tfs)
            l1 = tf.layers.dense(l0, 200, tf.nn.relu, trainable=trainable)
            l10 =  tf.layers.batch_normalization(l1)
            l2 = tf.layers.dense(l10, 100, tf.nn.relu, trainable=trainable)
            l20 =  tf.layers.batch_normalization(l2)

            mu = tf.layers.dense(l20, A_DIM, tf.nn.tanh, trainable=trainable, name = 'mu_'+name)
            sigma= tf.layers.dense(l20, A_DIM, tf.nn.softplus, trainable=trainable,name ='sigma_'+name )
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma) # Loc is the mean
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) #Collect the weights of the layers l1,mu/2,sigma
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        #self.sess.run(self.update_states)
        #print(a)
        return np.clip(a, *ACTION_BOUND) # limits the output of values between -1 & 1, to each of the values of 'a'

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0] # departure from NN del Critic|| V = learned state-value function
    


sess = tf.Session()
ppo = PPO(sess,LOAD)
path = './models'
saver = tf.train.Saver()