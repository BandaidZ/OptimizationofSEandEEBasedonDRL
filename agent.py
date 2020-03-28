from __future__ import print_function, division
import os
import time
import random
import numpy as np
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'
        self.env = environment
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir)
        self.max_step = 100000
        # The number of RB, The number of vehicle
        self.RB_number = 20
        self.num_vehicle = len(self.env.vehicles)
        # The following two variables are used to store the transmission power 
        # and channel selection of each V2V link
        # The one is used for testing, and the other is used for training
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],
                                              dtype='int32')  # this is actions that taken by V2V links with power
                                                                # what's the 2 meaning ?
        self.action_all_with_power_training = np.zeros([self.num_vehicle, 3, 2],
                                                       dtype='int32')
        self.reward = []
        # Settings related to learning rate
        self.learning_rate = 0.01    # 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        # each 100 steps update the target_q network
        self.target_q_update_step = 100  # 100
        #Discount factor
        self.discount = 0.5
        self.double_q = True
        self.build_dqn()
        # The number of V2V links.
        self.V2V_number = 3 * len(self.env.vehicles)  # every vehicle need to communicate with 3 neighbors
        self.training = True

    # This function is used to store the transmit power and channel selected by each V2V link 
    # Store in an <"action"> matrix 
    def merge_action(self, idx, action):    # don't know
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action / self.RB_number))
   
    def get_state(self, idx):
        # ===============================
        #  Get State from the environment
        # ===============================
        vehicle_number = len(self.env.vehicles)
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0], self.env.vehicles[idx[0]].destinations[idx[1]],
                       :] - 80) / 60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
        Eve_channel_I = (self.env.Eve_channels_with_fastfading_I[idx[0], :] - 80) / 60
        Eve_channel_V = (self.env.Eve_channels_with_fastfading_V[idx[0], self.env.vehicles[idx[0]].destinations[idx[1]],
                       :] - 80) / 60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
        # The <"NeiSelection"> representative RB occupation
        NeiSelection = np.zeros(self.RB_number)
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[self.action_all_with_power_training[self.env.vehicles[idx[0]].neighbors[i], j, 0]] = 1
                else:
                    NeiSelection[self.action_all_with_power[self.env.vehicles[idx[0]].neighbors[i], j, 0]] = 1
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0], i, 0] >= 0:
                    NeiSelection[self.action_all_with_power_training[idx[0], i, 0]] = 1
            else:
                if self.action_all_with_power[idx[0], i, 0] >= 0:
                    NeiSelection[self.action_all_with_power[idx[0], i, 0]] = 1
        # Status include V2I_channel, V2V_interference, V2V_channel, Eve_channel_I, Eve_channel_V, NeiSelection
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, Eve_channel_I, Eve_channel_V, NeiSelection))

    def predict(self, s_t, step, test_ep=False):
        # ==========================
        #  Select actions
        # ==========================
        ep = 1 / (step / 1000000 + 1)
        # Random selection or training selection
        if random.random() < ep and test_ep is False:  # epsion to balance the exporation and exploition
            # Each number from 0 ~ 60 represents a choice
            action = np.random.randint(60)  # 20RBs X 3 power level
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]   # ?
        return action

    # This function used for collcet data for training, and training a mini batch
    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 用于experience replay
        # ---------
        self.memory.add(prestate, state, reward, action)  # add the state and the action and the reward to the memory
        # print(self.step)
        if self.step > 0:
            if self.step % 50 == 0:
                # print('Training')
                self.q_learning_mini_batch()  # training a mini batch
                # self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                # print("Update Target Q network:")
                self.update_target_q_network()  # 更新目标Q网络的参数

    # The network training and testing funtion
    def train(self):    # 要修改
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        print(self.num_vehicle)
        #!Step1: Start a new simulation environment
        self.env.new_random_game(self.num_vehicle)   # 感觉相当于episode
        for self.step in (range(0, 40000)):  # need more configuration
            #!Step2: Begin training, the tutal steps is 40000
            # initialize set some varibles
            if self.step == 0:  
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []
            # Restart a new simulation environment
            if (self.step % 2000 == 1):
                self.env.new_random_game(self.num_vehicle)
            print(self.step)
            state_old = self.get_state([0, 0])
            # print("state", state_old)
            self.training = True
            for k in range(1):
                for i in range(len(self.env.vehicles)):
                    for j in range(3):
                        #!Step3: Get training data for each pair of V2V links and training
                        # Include <"state_old, state_new, reward_train, action">
                        # Besides: The training a batch in <"observe"> function
                        state_old = self.get_state([i, j])
                        action = self.predict(state_old, self.step)
                        # self.merge_action([i,j], action)
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action / self.RB_number))
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i, j])
                        state_new = self.get_state([i, j])
                        self.observe(state_old, state_new, reward_train, action)
            if (self.step % 2000 == 0) and (self.step > 0):
                #!Step4: Testing
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50
                if (self.step == 38000):
                    number_of_game = 100
                V2V_Eifficency_list = np.zeros(number_of_game)
                V2I_Eifficency_list = np.zeros(number_of_game)
                V2V_security_rate_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Eifficency_V2V = []
                    Eifficency_V2I = []
                    Security_rate = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i, :, 0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                            for j in sorted_idx:
                                state_old = self.get_state([i, j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i, j], action)
                            if i % (len(self.env.vehicles) / 10) == 1:  # 都加10次？
                                action_temp = self.action_all_with_power.copy()
                                V2V_reward, V2I_reward, V2V_security_rate = self.env.act_asyn(action_temp)  # self.action_all)
                                Eifficency_V2V.append(np.sum(V2V_reward))
                                Eifficency_V2I.append(np.sum(V2I_reward))
                                Security_rate.append(np.sum(V2V_security_rate))
                        # print("actions", self.action_all_with_power)
                    V2V_Eifficency_list[game_idx] = np.mean(np.asarray(Eifficency_V2V))
                    V2I_Eifficency_list[game_idx] = np.mean(np.asarray(Eifficency_V2I))
                    V2V_security_rate_list[game_idx] = np.mean(np.asarray(Security_rate))
                    # print("action is", self.action_all_with_power)
                    # print('failure probability is, ', percent)
                    # print('action is that', action_temp[0,:])
                #!Step5: Save weight parameters
                self.save_weight_to_pkl()
                print('The number of vehicle is ', len(self.env.vehicles))
                print('Mean of the V2V Eifficency is that ', np.mean(V2V_Eifficency_list))
                print('Mean of the V2I Eifficency is that ', np.mean(V2I_Eifficency_list))
                print('Mean of V2V Security Rate is that ', np.mean(V2V_security_rate_list))
                # print('Test Reward is ', np.mean(test_result))

    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        # s_t, action,reward, s_t_plus_1, terminal = self.memory.sample()
        s_t, s_t_plus_1, action, reward = self.memory.sample()
        # print()
        # print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])
        t = time.time()
        if self.double_q:  # double Q learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1,
                                                                       self.target_q_idx: [[idx, pred_a] for idx, pred_a
                                                                                           in enumerate(pred_action)]})
            target_q_t = self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 + reward
        _, q_t, loss, w = self.sess.run([self.optim, self.q, self.loss, self.w],
                                        {self.target_q_t: target_q_t, self.action: action, self.s_t: s_t,
                                         self.learning_rate_step: self.step})  # training the network

        print('loss is ', loss)  # 每喂一批数据更新一次loss function
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def build_dqn(self):
        # --- Building the DQN -------
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 120
        n_output = 60
        # The DQN network weights and biases
        def encoder(x):
            weights = {
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),

            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return layer_4, weights
        # Used for prediction
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, n_input])
            self.q, self.w = encoder(self.s_t)
            self.q_action = tf.argmax(self.q, dimension=1)
        # Used for get target-Q
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
        # Used for update the target-Q network parameters
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])

        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        # Used for Optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32', None, name='action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(self.learning_rate, self.learning_rate_step,
                                                                          self.learning_rate_decay_step,
                                                                          self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(
                self.loss)

        tf.initialize_all_variables().run()
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    # These two functions are used to save and load weight parameters
    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network()
