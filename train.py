"""
implement a CNN network as mentioned in VIN paper.
Author: kenneth yu
"""

import tensorflow as tf
import os
import time
import glob
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import variance_scaling_initializer, l2_regularizer
from cnn_model import CNNModel
from input import GridDomainReader, TRAJ_TRAIN_DATA_FILENAME,TRAJ_TEST_DATA_FILENAME, action_rc_diff

TRAINING_CFG = tf.app.flags.FLAGS  # alias
_flags = tf.app.flags

# static configuration of vin_tf model
_flags.DEFINE_integer('random_seed', 187, 'random seed')
_flags.DEFINE_string('summary_keys', 'summary_keys', '')
_flags.DEFINE_string('log_summary_keys', 'log_summary_keys', '')
_flags.DEFINE_integer('max_domains', 5000, 'maximum domains')
_flags.DEFINE_integer('action_dims', 8, 'action dims')

_flags.DEFINE_float('initial_learning_rate', 2.5e-3,
                    'Initial learning rate.')
_flags.DEFINE_float('num_epochs_per_decay', 20.0,
                    'Epochs after which learning rate decays.')
_flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                    'Learning rate decay factor.')
# _flags.DEFINE_float('learning_rate', 2.5e-4, '')
_flags.DEFINE_bool('is_training', 'True', '')

# hyper-P
if True:  # for test only
	_flags.DEFINE_float('reg_ratio', 1e-5, '')
	_flags.DEFINE_integer('num_training_steps', 1000, 'max training steps')  # 1M steps total
	_flags.DEFINE_integer('summary_freq', 30, '')
	_flags.DEFINE_integer('eval_freq', 1000, 'eval during training per steps')
	_flags.DEFINE_integer('num_eval_steps', 50, 'eval steps')
	_flags.DEFINE_integer('batch_size', 32, 'batch_size')
	_flags.DEFINE_integer('l_r_decay_freq', 5 * (10 ** 5), 'decay freq')  # 500k
	_flags.DEFINE_string('log_dir', './log_dir/', '')
	_flags.DEFINE_string('checkpoint_dir', './chk_pnt/', '')
	_flags.DEFINE_string('train_data_filename', './data/gridworld_28.mat', 'gridworld matlab train data file')
	_flags.DEFINE_string('test_data_filename', './data/gridworld_28_test.mat', 'gridworld matlab test data file')
	_flags.DEFINE_multi_integer('im_size', [28, 28], 'image dimensions')
else:
	_flags.DEFINE_float('reg_ratio', 1e-5, '')
	_flags.DEFINE_integer('num_training_steps', 10 * (10 ** 5), '')  # 1M steps total
	_flags.DEFINE_integer('summary_freq', 1 * 10000, '')  # 10~min
	_flags.DEFINE_integer('summary_transition_freq', 1 * 10000, '')
	_flags.DEFINE_integer('eval_freq', 3 * 10000, 'eval during training per steps')
	_flags.DEFINE_integer('num_eval_steps', 1000, 'eval steps')
	_flags.DEFINE_integer('batch_size', 64, 'batch_size')
	_flags.DEFINE_integer('l_r_decay_freq', 5 * (10 ** 5), 'decay freq')  # 500k
	_flags.DEFINE_string('log_dir', './log_dir', '')
	_flags.DEFINE_string('checkpoint_dir', './chk_pnt', '')
	_flags.DEFINE_string('train_data_filename', './data/gridworld_28.mat', 'gridworld matlab train data file')
	_flags.DEFINE_string('test_data_filename', './data/gridworld_28_test.mat', 'gridworld matlab test data file')
	_flags.DEFINE_multi_integer('im_size', [28, 28], 'image dimensions')

'''
normal structure for each conv layer:conv -> elu -> bn -> pooling.

conv-1: 3x3 s:1, 50 channels .no padding .
max pooling: 2x2
conv-2: 3x3,s:1, 50 channels. no padding .
conv-3: 3x3,s:1. 100 channels. no padding .
max pooling: 2x2
conv-4: 3x3 s:1, 100 channels. no padding.
conv-5: 3x3 s:1, 100 channels. no padding.

fc-1:200-units. followed  by elu.
fc-2: 4-units. output is logits.

output: unscaled logits of each actions:
N,S,E,W,NE,NW,SE,SW

=== state space:
	s_image of grid map (1:Obstrcle, 0:free space)
	s_goal (goal: 10, other: 0)
	s_curr_pos of current state (1: curr pos, 0: others.)

'''


class CNNModelCfg:
	def __init__(self):
		# === net arch
		# -- 5 conv layers --
		self.conv_n_feature_maps = [50, 50, 100, 100, 100]  # number of filters
		self.conv_kernel_sizes = [(3, 3)] * 5
		self.conv_strides = [1] * 5
		self.conv_paddings = ['SAME'] * 5
		self.pooling_kernel_sizes = [(2, 2), None, (2, 2), None, None]
		self.pooling_strides = [(2, 2), None, (2, 2), None, None]
		self.conv_activations = [tf.nn.elu] * 5
		self.conv_dropout_keep_probs = [1, 0.6, 1, 1, 0.6]
		self.conv_initializers = [variance_scaling_initializer()] * 5  # [He] init.
		###TODO try w/o BN
		# _flags.DEFINE_.conv_normalizers = [batch_norm] * 5
		# _flags.DEFINE_.conv_normal_params = [{'is_training':is_training,
		#                                        'data_format':'NHWC=
		#                                        'updates_collections':None }] *3 # inplace update running average
		self.conv_normalizers = [None] * 5
		self.conv_norm_params = [{}] * 5
		self.conv_regularizers = [l2_regularizer(scale=TRAINING_CFG.reg_ratio)] * 5
		# -- 2 fc layers including output layer--
		self.n_fc_units = [200, TRAINING_CFG.action_dims]
		self.fc_activations = [tf.nn.elu, None]
		self.fc_dropout_keep_probs = [0.6, 1]
		self.fc_initializers = [variance_scaling_initializer()] * 2  # [He] init
		self.fc_regularizers = [l2_regularizer(scale=TRAINING_CFG.reg_ratio), None]
		###TODO try w/o BN
		# _flags.DEFINE_.fc_normalizers = [batch_norm, None]  # 2nd fc including action input and no BN
		# _flags.DEFINE_.fc_norm_params =  [{'is_training':is_training,
		#                                        'data_format':'NHWC',
		#                                        'updates_collections':None }, None] # inplace update running average
		self.fc_normalizers = [None] * 2  # 2nd fc including action input and no BN
		self.fc_norm_params = [{}] * 2


# global var for eval time counting
prev_eval_time = 0.0

tf.logging.set_verbosity(tf.logging.INFO)


class Trainer(object):
	def __init__(self):
		np.random.seed(TRAINING_CFG.random_seed)
		# we input 3 channels data: obstacle_img, goal_image, curr_position. dtype is int8. NHWC format.
		self.x_inputs = tf.placeholder(tf.int8, shape=(None, TRAINING_CFG.im_size[0], TRAINING_CFG.im_size[1], 3))
		self.state_inputs = tf.cast(self.x_inputs, tf.float32)
		# label value is action idx:0-7
		self.y_inputs = tf.placeholder(tf.int8, shape=(None,))
		self.labels = tf.cast(self.y_inputs, tf.int32)  # softmax op requires int32.
		## track updates.
		self.global_step_tensor = tf.train.create_global_step()
		self.is_training = tf.placeholder(tf.bool, shape=())
		# dataset
		self.data_reader = GridDomainReader(TRAINING_CFG.train_data_filename, TRAINING_CFG.test_data_filename,TRAINING_CFG.im_size,	TRAINING_CFG.batch_size)

		num_batches_per_epoch = (self.data_reader.num_examples_per_epoch('train') / TRAINING_CFG.batch_size)
		# Decay steps need to be divided by the number of replicas to aggregate.
		decay_steps = num_batches_per_epoch * TRAINING_CFG.num_epochs_per_decay

		# Decay the learning rate exponentially based on the number of steps.
		# not create variable. staircase.
		self.l_r = tf.train.exponential_decay(TRAINING_CFG.initial_learning_rate,self.global_step_tensor, decay_steps,TRAINING_CFG.learning_rate_decay_factor,staircase=True)
		# Add a summary to track the learning rate tensor.
		tf.summary.scalar('learning_rate', self.l_r, collections=[TRAINING_CFG.summary_keys])

		optimizer = tf.train.AdamOptimizer(learning_rate=self.l_r)

		cnn_model_cfg = CNNModelCfg()
		self.model = CNNModel(cnn_model_cfg, optimizer, is_training=self.is_training)
		self.train_op, self.loss_mean_tensor, self.predicted_classes = self.model.create_net(self.state_inputs,self.labels, self.global_step_tensor)

		self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

		# for summary text
		self.summary_text_tensor = tf.convert_to_tensor(str('summary_text'), preferred_dtype=str)
		tf.summary.text(name='summary_text', tensor=self.summary_text_tensor, collections=[TRAINING_CFG.log_summary_keys])

		##create tf default session
		# TODO any session option to control gpu card usage?
		self.sess = tf.Session(graph=tf.get_default_graph())
		# note: will transfer graph to graphdef now. so we must finish all the computation graph
		# before creating summary writer.
		self.summary_writer = tf.summary.FileWriter(logdir=os.path.join(TRAINING_CFG.log_dir, "train"), graph=self.sess.graph)
		self.summary_op = tf.summary.merge_all(key=TRAINING_CFG.summary_keys)
		self.log_summary_op = tf.summary.merge_all(key=TRAINING_CFG.log_summary_keys)

	def train(self):
		######### initialize computation graph  ###########
		self.sess.run(fetches=[tf.global_variables_initializer()])

		# Load a previous checkpoint if it exists
		latest_checkpoint = tf.train.latest_checkpoint(TRAINING_CFG.checkpoint_dir)
		if latest_checkpoint:
			print("=== Loading model checkpoint: {}".format(latest_checkpoint))
			self.saver.restore(self.sess, latest_checkpoint)

		####### start training #########
		n_episodes = 1
		update_start = 0.0

		for step in range(0, TRAINING_CFG.num_training_steps):
			if step < 10:
				update_start = time.time()

			x_batch, y_batch = self.data_reader.batch_train_inputs()

			# run_options = tf.RunOptions(output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE)
			if 0 == step % TRAINING_CFG.summary_freq:
				# run_metadata = tf.RunMetadata()
				_, summary, loss_mean_value, global_step_value = self.sess.run(fetches=[self.train_op, self.summary_op,
				                                                                        self.loss_mean_tensor,
				                                                                        self.global_step_tensor],
				                                                               feed_dict={self.x_inputs: x_batch,
				                                                                          self.y_inputs: y_batch,
				                                                                          self.is_training: True
				                                                                          })

				self.summary_writer.add_summary(summary, global_step_value)
				self.summary_writer.flush()
			else:
				_, loss_mean_value = self.sess.run(fetches=[self.train_op, self.loss_mean_tensor],
				                                   feed_dict={self.x_inputs: x_batch,
				                                              self.y_inputs: y_batch,
				                                              self.is_training: True
				                                              })

			# if step % 2000 == 0:
			if step % 20 == 0:
				tf.logging.info('@@ step:{} total_loss_mean:{}'.format(step, loss_mean_value))

			# --end of summary --

			# test update duration at first 10 update
			if step < 10:
				tf.logging.info(' @@@@ one batch learn duration @@@@:{}'.format(time.time() - update_start))

			def estimate_fn():
				x_val_batch, y_val_batch = self.data_reader.batch_validation_inputs()
				return self.sess.run(fetches=[self.loss_mean_tensor],
				                     feed_dict={self.x_inputs: x_val_batch,
				                                self.y_inputs: y_val_batch,
				                                self.is_training: False
				                                })[0]

			if step % TRAINING_CFG.eval_freq == 0:
				self.evaluate(TRAINING_CFG.num_eval_steps, estimate_fn)

	###### end of train #######

	def evaluate(self, num_eval_steps, estimate_fn):
		total_loss = 0.0
		max_loss = 0.0
		min_loss = 1e3
		global_step = self.sess.run(fetches=[self.global_step_tensor])[0]
		tf.logging.info(' ####### start evaluate @ step:{}##  '.format(global_step))
		for step in range(0, num_eval_steps):
			loss = estimate_fn()
			max_loss = max(max_loss, loss)
			min_loss = min(min_loss, loss)
			total_loss += loss

		avg_loss = total_loss / num_eval_steps
		# we always save model each evaluation.
		saved_name = self.save_model()
		self.write_summary(global_step, avg_loss, max_loss, min_loss, saved_name)
		tf.logging.info(
			'@@@@@@ eval saving model : global_step:{} - avg_loss:{} - max_loss:{} - min_loss:{} - saved_file: {} @@@@@@ '.format(
				global_step,
				avg_loss, max_loss, min_loss, saved_name))

	def save_model(self):
		saved_name = self.saver.save(sess=self.sess, save_path=TRAINING_CFG.checkpoint_dir,
		                             global_step=self.global_step_tensor)
		for _ in range(10):  # wait total 10s
			if not glob.glob(saved_name + '*'):
				time.sleep(1.0)
			else:
				return saved_name
		raise FileNotFoundError('@@@@@@@@@@@@ save model failed: {}'.format(saved_name))

	def write_summary(self, global_step, avg_loss, max_loss, min_loss, saved_name):
		eval_summary = tf.Summary()  # proto buffer
		eval_summary.value.add(node_name='avg_loss', simple_value=avg_loss, tag="train_eval/avg_loss")
		eval_summary.value.add(node_name='max_loss', simple_value=max_loss, tag="train_eval/max_loss")
		eval_summary.value.add(node_name='min_loss', simple_value=min_loss, tag="train_eval/min_loss")
		self.summary_writer.add_summary(summary=eval_summary, global_step=global_step)

		# write log info to summary
		log_info = 'eval save model : global_step:{} avg_loss:{} \
				max_loss:{}  min_loss:{}  saved_file: {} '.format( \
			global_step, avg_loss, max_loss, min_loss, saved_name)
		log_summary = self.sess.run(fetches=[self.log_summary_op],
		                            feed_dict={self.summary_text_tensor: log_info})
		self.summary_writer.add_summary(summary=log_summary[0], global_step=global_step)
		self.summary_writer.flush()

	# TODO move into env
	@staticmethod
	def excute_action(curr_state_rc, action, goal_rc, im):
		next_state_rc = curr_state_rc + action_rc_diff[action]
		failed = False
		term = False
		# out of border check
		if np.any(next_state_rc >= TRAINING_CFG.im_size) or np.any(next_state_rc < [0, 0]):
			failed = True
			term = True
		# obstacle:
		elif im[next_state_rc[0], next_state_rc[1]] != 0:
			failed = True
			term = True
		# bingo!
		elif np.all(next_state_rc == goal_rc):
			failed = False
			term = True
		# normal step
		else:
			failed = False
			term = False

		return failed, term, next_state_rc

	@staticmethod
	def np_array_add_one_dim(arr):
		return np.reshape(arr, (1, *arr.shape))

	def predict_one_batch(self, x_inputs):
		return self.sess.run(fetches=[self.predicted_classes],
		                     feed_dict={self.x_inputs: x_inputs,
		                                self.is_training: False
		                                })

	def predict_and_show(self, filename=TRAJ_TEST_DATA_FILENAME):
		"""
		predict and compare with expert data.
		:param filename:
		:return:
		"""
		######### initialize computation graph  ###########
		# self.sess.run(fetches=[tf.global_variables_initializer()])

		# Load a previous checkpoint if it exists
		latest_checkpoint = tf.train.latest_checkpoint(TRAINING_CFG.checkpoint_dir)
		if latest_checkpoint:
			print("=== Loading model checkpoint: {}".format(latest_checkpoint))
			self.saver.restore(self.sess, latest_checkpoint)

		f = open(filename, 'rb')
		pred_traj = None
		# TODO use generator to yield traj one by one.
		while True:  # load every expert traj to compare
			try:
				traj_record = pkl.load(f)
			except e:
				break
			[im, expert_traj, goal_rc] = traj_record.im, traj_record.trajectory_rc, traj_record.goal_rc
			curr_state_rc = expert_traj[0]  # start position
			pred_traj = None
			term = False
			failed = False

			for _ in range(100):  # rollout
				x_data = GridDomainReader.make_x_data_batch(self.np_array_add_one_dim(im),
				                                            self.np_array_add_one_dim(goal_rc),
				                                            self.np_array_add_one_dim(curr_state_rc))
				action = self.predict_one_batch(x_data)[0][0]
				failed, term, curr_state_rc = self.excute_action(curr_state_rc, action, goal_rc, im)
				# we always record pred_traj even when next state failed.
				if pred_traj is None:  # new pred_traj
					pred_traj = self.np_array_add_one_dim(curr_state_rc)  # add 1 dim for append
				else:
					pred_traj = np.append(pred_traj, self.np_array_add_one_dim(curr_state_rc),axis=0)
				if term:  # arrival or into obstacle/out_of_border
					break

			if (not term) or failed:
				# TODO statistics
				tf.logging.info(' ==== failed to predict traj ==== ')
			# we always plot expert traj and pred traj
			# TODO calc diff
			plt.figure(1)
			plt.imshow(im, cmap=plt.cm.gray)
			plt.plot(expert_traj[:, 1], expert_traj[:, 0],label = 'expert path',color='blue',marker='x',linewidth=1)
			plt.plot(pred_traj[:, 1], pred_traj[:, 0], label='predict path',color='red', marker = '*',linewidth=1)
			# plt.legend(['expert path:', 'predict path:'])
			# plot goal
			plt.plot(goal_rc[1], goal_rc[0], label='goal', color='green',marker='o')  # in pyplot, row -> plot coord-y.
			plt.plot(expert_traj[0][1], expert_traj[0][0], label='start',color='blue',marker='s')  # start point
			plt.legend()
			plt.show(1)

		f.close()
