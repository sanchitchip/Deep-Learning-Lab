import os
import pickle
import tensorflow as tf
import gym
import numpy as np
import cv2
from gym import spaces
from collections import deque
import argparse
from matplotlib import pyplot as plt
import time
import pdb
from google.colab import drive


class NoopResetEnv(gym.Wrapper):
	def __init__(self, env, noop_max=30):
		"""Sample initial states by taking random number of no-ops on reset.
		No-op is assumed to be action 0.
		"""
		gym.Wrapper.__init__(self, env)
		self.noop_max = noop_max
		self.override_num_noops = None
		if isinstance(env.action_space, gym.spaces.MultiBinary):
			self.noop_action = np.zeros(self.env.action_space.n, dtype=np.int64)
		else:
			# used for atari environments
			self.noop_action = 0
			assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

	def _reset(self, **kwargs):
		""" Do no-op action for a number of steps in [1, noop_max]."""
		self.env.reset(**kwargs)
		if self.override_num_noops is not None:
			noops = self.override_num_noops
		else:
			noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
		assert noops > 0
		obs = None
		for _ in range(noops):
			obs, _, done, _ = self.env.step(self.noop_action)
			if done:
				obs = self.env.reset(**kwargs)
		return obs


class FireResetEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
		assert len(env.unwrapped.get_action_meanings()) >= 3

	def _reset(self, **kwargs):
		self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(1)
		if done:
			self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(2)
		if done:
			self.env.reset(**kwargs)
		return obs


class EpisodicLifeEnv(gym.Wrapper):
	def __init__(self, env):
		"""Make end-of-life == end-of-episode, but only reset on true game over.
		Done by DeepMind for the DQN and co. since it helps value estimation.
		"""
		gym.Wrapper.__init__(self, env)
		self.lives = 0
		self.was_real_done  = True

	def _step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		# check current lives, make loss of life terminal,
		# then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			# for Qbert somtimes we stay in lives == 0 condtion for a few frames
			# so its important to keep lives > 0, so that we only reset once
			# the environment advertises done.
			done = True
		self.lives = lives
		return obs, reward, done, info

	def _reset(self, **kwargs):
		"""Reset only when lives are exhausted.
		This way all states are still reachable even though lives are episodic,
		and the learner need not know about any of this behind-the-scenes.
		"""
		if self.was_real_done:
			obs = self.env.reset(**kwargs)
		else:
			# no-op step to advance from terminal/lost life state
			obs, _, _, _ = self.env.step(0)
		self.lives = self.env.unwrapped.ale.lives()
		return obs


class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env, skip=4):
		"""Return only every `skip`-th frame"""
		gym.Wrapper.__init__(self, env)
		# most recent raw observations (for max pooling across time steps)
		self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype='uint8')
		self._skip       = skip

	def _step(self, action):
		"""Repeat action, sum reward, and max over last observations."""
		total_reward = 0.0
		done = None
		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			if i == self._skip - 2: self._obs_buffer[0] = obs
			if i == self._skip - 1: self._obs_buffer[1] = obs
			total_reward += reward
			if done:
				break
		# Note that the observation on the done=True frame
		# doesn't matter
		max_frame = self._obs_buffer.max(axis=0)

		return max_frame, total_reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
	def _reward(self, reward):
		"""Bin reward to {+1, 0, -1} by its sign."""
		return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
	def __init__(self, env):
		"""Warp frames to 84x84 as done in the Nature paper and later work."""
		gym.ObservationWrapper.__init__(self, env)
		self.width = 84
		self.height = 84
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

	def _observation(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
		return frame[:, :, None]


class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

	def _reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	def _step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.k
		return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
	def _observation(self, observation):
		# careful! This undoes the memory optimization, use
		# with smaller replay buffers only.
		return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
	def __init__(self, frames):
		"""This object ensures that common frames between the observations are only stored once.
		It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
		buffers.
		This object should only be converted to numpy array before being passed to the model.
		You'd not belive how complex the previous solution was."""
		self._frames = frames

	def __array__(self, dtype=None):
		out = np.concatenate(self._frames, axis=2)
		if dtype is not None:
			out = out.astype(dtype)
		return out


def make_atari(env_id):
	env = gym.make(env_id)
	assert 'NoFrameskip' in env.spec.id
	env = NoopResetEnv(env, noop_max=30)
	env = MaxAndSkipEnv(env, skip=4)
	return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
	"""Configure environment for DeepMind-style Atari.
	"""
	if episode_life:
		env = EpisodicLifeEnv(env)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = FireResetEnv(env)
	env = WarpFrame(env)
	if scale:
		env = ScaledFloatFrame(env)
	if clip_rewards:
		env = ClipRewardEnv(env)
	if frame_stack:
		env = FrameStack(env, 4)
	return env


def wrap_atari_deepmind(env_id='Breakout-v0', clip_rewards=True):

	#env = gym.make(env_id)
	env = make_atari(env_id)
	return wrap_deepmind(env, clip_rewards=clip_rewards, frame_stack=True, scale=True)



##### IMPORTANT DONT FORGET TO COPY THE PARAMETERS OF ONLINE NETWORK TO TARGET NETWORK in the init state although won;t
##### affect the performance but might deduct some marks for it            
class Breakout_DQN(object):
    def __init__(self, env,env_test,C):

      tf.reset_default_graph()
      self.env = env
      self.env_test = env_test
      self.loss_lis = []
      self.loss_step=[]
      self.num_action = self.env.action_space.n
      self.feat_shp = self.env.observation_space.shape
      # will be needed in when extracting elements from memory buffer and storing in memory buffer also. 
      self.num_feat = np.prod(np.array(self.feat_shp))
      self.memory_size = 10000
      # memory array initialized with zeros with ever row has a length of 
      #2*(element in state) as there will be current and next state and 3 element for reward,omega and action taken.
      self.memory = np.zeros((self.memory_size, 3 + self.num_feat*2)) 
      self.reward_lis = []
      ## for evaluation.
      self.total_ep_plays = 5
      self.n_plays=30
      self.play_reward=[]
      self.play_steps = []
    # training settings
      self.num_step = 2e6
      self.eval_steps = 1e5
  # uncomment these for bonus question    
 #     self.num_step = 3e5
 #     self.eval_steps = 2e4
      self.param_update_every = 4
      self.C = C
      self.init_learning = 10000
      self.gamma = 0.99
      self.batch_size = 32
      self.epsilon_min = 0.1 
      self.epsilon = 1.0
      self.eps_reduce_by = (self.epsilon - self.epsilon_min) / (1000000)
      self.optim = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99)
      self.epsilon_test = .001
      self.mem_count = 0
      # model
      self.model_skelton()
      online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
      tar_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
      with tf.variable_scope('param_copy'):
        self.target_copy = [tf.assign(t, o) for t, o in zip(tar_params, online_params)]

      # tf initialization
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      # bonus2 statement:SET THIS TO TRUE ONLY IF YOU WANT TO CHECK THE BONUS QUESTION AND HAVE THE .npy file in the directory.
      self.bonus2=False

		
    def model_skelton(self):

      from tensorflow.python.ops.init_ops import VarianceScaling
      weight_init = VarianceScaling(scale=2,distribution='truncated_normal')
      bias_init = tf.zeros_initializer()

      vTensor_shape = [None] + [i for i in self.feat_shp] 
      self.curr_s = tf.placeholder(tf.float32, vTensor_shape, name='current_state') 
      self.next_state = tf.placeholder(tf.float32, vTensor_shape, name='next_state') 
      # should change
      self.q_target = tf.placeholder(tf.float32, [None, self.num_action], name='q_t') 

      with tf.variable_scope('eval_net'):
        e_conv1 = tf.layers.conv2d(
            inputs=self.curr_s,
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='e_conv1')
        e_conv2 = tf.layers.conv2d(
            inputs=e_conv1,
            filters=64,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='e_conv2')
        e_conv3 = tf.layers.conv2d(
            inputs=e_conv2,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='e_conv3')
        e_flat = tf.contrib.layers.flatten(e_conv3)
        e_dense1 = tf.layers.dense(
            inputs=e_flat,
            units=512, 
            activation=tf.nn.relu,
            kernel_initializer=weight_init,
            bias_initializer=bias_init, 
            name='e_dense1')
        self.q_new = tf.layers.dense(
            inputs=e_dense1, 
            units=self.num_action,
            activation=None,
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            name='q_new') # q_new shape: (batch_size, num_action)

      # ------------------ build target_net ------------------ #
      with tf.variable_scope('target_net'):
        t_conv1 = tf.layers.conv2d(
            inputs=self.next_state,
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='t_conv1')
        t_conv2 = tf.layers.conv2d(
            inputs=t_conv1,
            filters=64,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='t_conv2')
        t_conv3 = tf.layers.conv2d(
            inputs=t_conv2,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu6,
            kernel_initializer=weight_init,
            name='t_conv3')
        t_flat = tf.contrib.layers.flatten(t_conv3)
        t_dense1 = tf.layers.dense(
            inputs=t_flat,
            units=512, 
            activation=tf.nn.relu,
            kernel_initializer=weight_init,
            bias_initializer=bias_init, 
            name='t_dense1')
        self.q_old = tf.layers.dense(
            inputs=t_dense1, 
            units=self.num_action,
            activation=None,
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            name='q_old')
      
      with tf.variable_scope('loss_computation'):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_new, name='TD_error'))

      with tf.variable_scope('optimization_Step'):
        self.train_op = self.optim.minimize(self.loss)

    def replay_buffer(self, s, a, r, d, s_):
      curr_state = np.reshape(s, [-1])
      next_state = np.reshape(s_, [-1])
      # arranging the 5 object as one array of length 84*84*4*2 [as there are two observations] + 3(reward,action,omega value)
      vArr = np.hstack((curr_state, [a, r, int(d)], next_state))
      ## replace with new memory at the particular index according to memory counter value
      index = self.mem_count % self.memory_size 
      self.memory[index, :] = vArr
      self.mem_count += 1

    def compute_loss(self):
      #made new changes here :: look out
#       pdb.set_trace()
      if self.mem_count > self.memory_size: sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False) 
      else: sample_index = np.random.choice(self.mem_count, size=self.batch_size)
      batch_memory = self.memory[sample_index, :]
      vTensor_shape = [self.batch_size] + [i for i in self.feat_shp] 
      ## reformating and generating the 5 object for batch size.

      curr_state = np.reshape(batch_memory[:, :self.num_feat], newshape=vTensor_shape)
      vAct = batch_memory[:, self.num_feat].astype(int) 
      rwd = batch_memory[:, self.num_feat + 1] 
      done = batch_memory[:, self.num_feat + 2] 
      next_state = np.reshape(batch_memory[:, -self.num_feat:], newshape=vTensor_shape)
      
      q_n, q_o = self.sess.run([self.q_new, self.q_old], feed_dict={ self.curr_s: curr_state, self.next_state: next_state })
      q_t = q_n.copy()
      batch_index = np.arange(self.batch_size, dtype=np.int32)


      sq_old = np.max(q_o, axis=1)

      q_t[batch_index, vAct] = rwd + (1-done) * self.gamma * sq_old 

      loss, _ = self.sess.run([ self.loss, self.train_op], feed_dict={ self.curr_s: curr_state, self.q_target: q_t })
      return loss


    def action_prediction(self, curr_obs,epsilon):
      #convert to (batch_size,84,84,4) bfore passing to network
      curr_obs = np.expand_dims(curr_obs, axis=0) 
      if (np.random.uniform() > epsilon):
        val = self.sess.run(self.q_new, feed_dict={self.curr_s: curr_obs}) 
        action = np.argmax(val)
      else:
        # use env.sample action function instead of this one.
        action = np.random.randint(0, self.num_action)      
      return action

    def train(self):
      # starting from random loss to avoid print statement problem and also avoid using extra if check conditions.
      vLoss = 5
      step = 0
      vEpisode = 0
      while step < self.num_step:
        # observation initialization        
        curr_obs = np.array(self.env.reset())
        done = False
        vEpisode_rwd = 0.0

        while not done:	
          action = self.action_prediction(curr_obs,self.epsilon) 
          obs_next, reward, done, info = self.env.step(action)
          vEpisode_rwd += reward

          self.replay_buffer(curr_obs, action, reward, done, np.array(obs_next))          
          if (self.bonus2==True) and (step==self.init_learning):
            drive.mount('/content/drive')
            vfile_name = "/content/drive/My Drive/replay_buffer_FOR_STUDENTS.npy"
            vTemp = np.load(vfile_name)
#            pdb.set_trace()
            print("Replay Buffer is loaded")
            self.memory = vTemp

          if (step >= self.init_learning) and (step % self.param_update_every == 0):
 #           s_t = time.time()
            vLoss = self.compute_loss()
            self.loss_lis.append(vLoss)
            self.loss_step.append(step)
#            e_t = time.time()
#            print("Loss computation time is {}".format(s_t-e_t))
          
          if (step >= self.init_learning) and (step % self.C == 0):
   #         s_t = time.time()
            self.sess.run(self.target_copy)
 #           e_t = time.time()
  #          print("Param copying time is {}".format(s_t-e_t))
          # eval step:::::to eval every 100,000 steps
          if (step%self.eval_steps==0) and (step>=self.init_learning):
#            s_t = time.time()
            vplay = self.eval_plays(step)            
            self.play_reward.append(vplay)            
            self.play_steps.append(step)
#            e_t = time.time()
#            print("Eval time is {}".format(s_t-e_t))                
          
          if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.eps_reduce_by
          else:
            self.epsilon_min 
          
          curr_obs = np.array(obs_next)
          step += 1
          
        vEpisode += 1
        self.reward_lis.append(vEpisode_rwd)
        if vEpisode%100==0:
          print('Step: %i/%i,  Episode: %i,  Action: %i,  Episode Reward: %.0f,  Epsilon: %.2f, Loss: %.5f' % (step, self.num_step, vEpisode, action, vEpisode_rwd, self.epsilon, vLoss))
      
      # storing evaluation rewards and the step number of play to make plot for problem 6      
      file_name = "play_reward_{}.npy".format(self.C)
      file_name2 = "play_label_{}.npy".format(self.C)
      np.save(file_name,self.play_reward)
      np.save(file_name2,self.play_steps)

			
			

    def eval_plays(self,step):
      rewards_test=[]
      for i in range(self.n_plays):
        play_reward=0
        for i in range(self.total_ep_plays):

          c_state = self.env_test.reset()
          episode_reward=0
          done=False
          while not done:
            action = self.action_prediction(c_state, self.epsilon_test)
            c_state, reward, done, info = self.env_test.step(action)
            episode_reward += reward
        
          play_reward+=episode_reward							

        rewards_test.append(play_reward)
          
      print("-------------------------------------EVALSTEP--------------------------")
      print("Eval Step")
      print("Len of playlist is {}".format(len(rewards_test)))
      vFinal_reward = np.mean(rewards_test)
      print(end='\r')
      print()
      print('Step: %i/%i,  Play Reward: %.0f' % (step, self.num_step, vFinal_reward))
      return vFinal_reward
            
    def plot(self):
      avg_rwd = []
      for i in range(len(self.reward_lis)):
        if i < 30:
          avg_rwd.append(np.mean(self.reward_lis[:i]))
        else:
          avg_rwd.append(np.mean(self.reward_lis[i-30:i]))
      plt.plot(np.arange(len(avg_rwd)), avg_rwd)
      plt.ylabel('Average Reward in Last 30 Episodes')
      plt.xlabel('Number of Episodes')
      plt.show()
        
    def plot_play(self):
      plt.plot(self.play_steps,self.play_reward)	
      plt.ylabel('Mean of Play Reward for 30 plays')
      plt.xlabel('Step Number')
      plt.show()
    
    def plot_TD(self):
      plt.plot(self.loss_step,self.loss_lis)
      plt.ylabel('TD Loss at every update')
      plt.xlabel('Number of Steps')
      plt.show()


def problem6():
  play_reward_C1 = np.load("play_reward_10000.0.npy")
  play_reward_C2 = np.load("play_reward_50000.0.npy")
  play_reward_label = np.load("play_label_10000.0.npy")
  plt.plot(play_reward_label,play_reward_C1,label="C=10,000")
  plt.plot(play_reward_label,play_reward_C2,label="C=50,000")
  plt.ylabel('Average Scores')
  plt.xlabel('Number of Steps')
  plt.legend(loc="upper left")  
  plt.show()

def Monitor_agent(agent,env): 
  file_name = "game_learned"
  env = gym.wrappers.Monitor(env,file_name, force=True)
  for i in range(15):
    c_state = env.reset()
    eps = 0.001
    done=False
    while not done:
      action = agent.action_prediction(c_state, eps)
      c_state, reward, done, info = env.step(action)
    
            




env_name = 'BreakoutNoFrameskip-v4'
env = wrap_atari_deepmind(env_name)
env_test = wrap_atari_deepmind(env_name,clip_rewards=False)
model = Breakout_DQN(env,env_test,C=1e4)
model.train()

#agent = Breakout_DQN(env,env_test,C=5e4)
#agent.train()

model.plot()
		
model.plot_play()
model.plot_TD()
