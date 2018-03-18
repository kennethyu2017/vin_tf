"""
implement a CNN network as mentioned in VIN paper.
Author: kenneth yu
"""

import tensorflow as tf
import os
import time
import glob
import numpy as np
import math as math
from tensorflow.contrib.layers import variance_scaling_initializer, l2_regularizer
from cnn_model import CNNModel
from input import GridDomainReader

TRAINING_CFG = tf.app.flags.FLAGS  # alias
_flags = tf.app.flags

# static configuration of vin_tf model
_flags.DEFINE_integer('random_seed', 187, 'random seed')
_flags.DEFINE_string('summary_keys', 'summary_keys','')
_flags.DEFINE_string('log_summary_keys', 'log_summary_keys','')
_flags.DEFINE_integer('max_domains',5000,'maximum domains')
_flags.DEFINE_integer('action_dims', 8, 'action dims')

_flags.DEFINE_float('initial_learning_rate', 2.5e-3,
                          'Initial learning rate.')
_flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
_flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')
# _flags.DEFINE_float('learning_rate', 2.5e-4, '')

# hyper-P
if True:  # for test only
    _flags.DEFINE_float('reg_ratio', 1e-5, '')
    _flags.DEFINE_integer('num_training_steps', 1000, 'max training steps')  # 1M steps total
    _flags.DEFINE_integer('summary_freq', 10, '')
    _flags.DEFINE_integer('eval_freq', 20, 'eval during training per steps')
    _flags.DEFINE_integer('num_eval_steps', 10, 'eval steps')
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
    self.conv_n_feature_maps= [50, 50, 100, 100, 100]  #number of filters
    self.conv_kernel_sizes= [(3, 3)] * 5
    self.conv_strides= [1] * 5
    self.conv_paddings= ['SAME'] * 5
    self.pooling_kernel_sizes= [(2, 2), None, (2, 2), None, None]
    self.pooling_strides= [(2, 2), None, (2, 2), None, None]
    self.conv_activations= [tf.nn.elu] * 5
    self.conv_dropout_keep_probs = [1, 0.6, 1, 1, 0.6]
    self.conv_initializers= [variance_scaling_initializer()] * 5  # [He] init.
    ###TODO try w/o BN
    # _flags.DEFINE_.conv_normalizers = [batch_norm] * 5
    # _flags.DEFINE_.conv_normal_params = [{'is_training':is_training,
    #                                        'data_format':'NHWC=
    #                                        'updates_collections':None }] *3 # inplace update running average
    self.conv_normalizers= [None] * 5
    self.conv_norm_params= [{}] * 5
    self.conv_regularizers= [l2_regularizer(scale=TRAINING_CFG.reg_ratio)] * 5
    # -- 2 fc layers including output layer--
    self.n_fc_units= [200, TRAINING_CFG.action_dims]
    self.fc_activations= [tf.nn.elu, None]
    self.fc_dropout_keep_probs = [0.6, 1]
    self.fc_initializers= [variance_scaling_initializer()] * 2  # [He] init
    self.fc_regularizers= [l2_regularizer(scale=TRAINING_CFG.reg_ratio), None]
    ###TODO try w/o BN
    # _flags.DEFINE_.fc_normalizers = [batch_norm, None]  # 2nd fc including action input and no BN
    # _flags.DEFINE_.fc_norm_params =  [{'is_training':is_training,
    #                                        'data_format':'NHWC',
    #                                        'updates_collections':None }, None] # inplace update running average
    self.fc_normalizers= [None] * 2  # 2nd fc including action input and no BN
    self.fc_norm_params= [{}] * 2


# global var for eval time counting
prev_eval_time = 0.0

tf.logging.set_verbosity(tf.logging.INFO)

def train():
    np.random.seed(TRAINING_CFG.random_seed)
    # we input 3 channels data: obstacle_img, goal_image, curr_position. dtype is int8. NHWC format.
    x_inputs = tf.placeholder(tf.int8, shape=(None, TRAINING_CFG.im_size[0], TRAINING_CFG.im_size[1], 3))
    state_inputs = tf.cast(x_inputs,tf.float32)
    # label value is action idx:0-7
    y_inputs = tf.placeholder(tf.int8, shape=(None))
    labels = tf.cast(y_inputs, tf.int32)  #softmax op requires int32.
    ## track updates.
    global_step_tensor = tf.train.create_global_step()

    is_training = tf.placeholder(tf.bool, shape=())

    # dataset
    data_reader = GridDomainReader(TRAINING_CFG.train_data_filename, TRAINING_CFG.test_data_filename,
                                   TRAINING_CFG.im_size,
                                   TRAINING_CFG.batch_size)

    num_batches_per_epoch = (data_reader.num_examples_per_epoch('train') /
                             TRAINING_CFG.batch_size)
    # Decay steps need to be divided by the number of replicas to aggregate.
    decay_steps = num_batches_per_epoch * TRAINING_CFG.num_epochs_per_decay

    # Decay the learning rate exponentially based on the number of steps.
    # not create variable. staircase.
    l_r = tf.train.exponential_decay(TRAINING_CFG.initial_learning_rate,
                                    global_step_tensor,
                                    decay_steps,
                                    TRAINING_CFG.learning_rate_decay_factor,
                                    staircase=True)
    # Add a summary to track the learning rate tensor.
    tf.summary.scalar('learning_rate', l_r, collections=[TRAINING_CFG.summary_keys])

    optimizer = tf.train.AdamOptimizer(learning_rate=l_r)

    cnn_model_cfg = CNNModelCfg()
    model = CNNModel(cnn_model_cfg,optimizer,is_training=is_training)
    train_op, loss_mean_tensor = model.create_net(state_inputs, labels, global_step_tensor)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

    # for summary text
    summary_text_tensor = tf.convert_to_tensor(str('summary_text'), preferred_dtype=str)
    tf.summary.text(name='summary_text', tensor=summary_text_tensor, collections=[TRAINING_CFG.log_summary_keys])

    ##create tf default session
    # TODO any session option to control gpu card usage?
    sess = tf.Session(graph=tf.get_default_graph())

    '''
  # note: will transfer graph to graphdef now. so we must finish all the computation graph
  # before creating summary writer.
  '''
    summary_writer = tf.summary.FileWriter(logdir=os.path.join(TRAINING_CFG.log_dir, "train"),
                                           graph=sess.graph)
    summary_op = tf.summary.merge_all(key=TRAINING_CFG.summary_keys)
    log_summary_op = tf.summary.merge_all(key=TRAINING_CFG.log_summary_keys)

    ######### initialize computation graph  ###########

    sess.run(fetches=[tf.global_variables_initializer()])

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(TRAINING_CFG.checkpoint_dir)
    if latest_checkpoint:
        print("=== Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    ####### start training #########
    n_episodes = 1
    update_start = 0.0

    for step in range(0, TRAINING_CFG.num_training_steps):
        if step < 10:
            update_start = time.time()

        x_batch, y_batch = data_reader.batch_train_inputs()

        # run_options = tf.RunOptions(output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE)
        if 0 == step % TRAINING_CFG.summary_freq:
            # run_metadata = tf.RunMetadata()
            _, summary, loss_mean_value,global_step_value = sess.run(fetches=[train_op, summary_op,
                                             loss_mean_tensor,global_step_tensor],
                                               feed_dict={x_inputs: x_batch,
                                                          y_inputs: y_batch,
                                                          is_training:True})

            summary_writer.add_summary(summary,global_step_value)
            summary_writer.flush()
        else:
            _, loss_mean_value = sess.run(fetches=[train_op, loss_mean_tensor],
                                      feed_dict={x_inputs: x_batch,
                                                 y_inputs: y_batch,
                                                 is_training:True})

        # if step % 2000 == 0:
        if step % 20 == 0:
            tf.logging.info('@@ step:{} total_loss_mean:{}'.format(step, loss_mean_value))

            # --end of summary --

        # test update duration at first 10 update
        if step < 10:
            tf.logging.info(' @@@@ one batch learn duration @@@@:{}'.format(time.time() - update_start))

        def estimate_fn():
          x_val_batch, y_val_batch = data_reader.batch_validation_inputs()
          return sess.run(fetches=[loss_mean_tensor],
                          feed_dict={x_inputs: x_val_batch,
                                     y_inputs: y_val_batch,
                                     is_training:False})[0]

        if step % TRAINING_CFG.eval_freq == 0:
          evaluate(TRAINING_CFG.num_eval_steps, estimate_fn,
                   summary_writer,saver, sess, log_summary_op,summary_text_tensor,global_step_tensor)
###### end of train #######


def evaluate(num_eval_steps, estimate_fn,summary_writer, saver, sess, log_summary_op, summary_text_tensor,global_step_tensor):
  total_loss = 0.0
  max_loss = 0.0
  min_loss = 1e3
  global_step = sess.run(fetches=[global_step_tensor])[0]
  tf.logging.info(' ####### start evaluate @ step:{}##  '.format(global_step))
  for step in range(0, num_eval_steps):
    loss = estimate_fn()
    max_loss = max(max_loss, loss)
    min_loss = min(min_loss, loss)
    total_loss += loss

  avg_loss = total_loss / num_eval_steps
  #we always save model each evaluation.
  saved_name = save_model(saver, sess, global_step_tensor)
  write_summary(summary_writer, global_step, avg_loss, max_loss, min_loss, saved_name, sess,log_summary_op,summary_text_tensor)
  tf.logging.info('@@@@@@ eval saving model : global_step:{} - avg_loss:{} - max_loss:{} - min_loss:{} - saved_file: {} @@@@@@ '.format(global_step,
                                                                          avg_loss,max_loss,min_loss, saved_name))


def save_model(saver, sess,global_step_tensor):
  saved_name=saver.save(sess=sess, save_path=TRAINING_CFG.checkpoint_dir, global_step=global_step_tensor)
  for _ in range(10): #wait total 10s
    if not glob.glob(saved_name + '*'):
      time.sleep(1.0)
    else:
      return saved_name
  raise FileNotFoundError('@@@@@@@@@@@@ save model failed: {}'.format(saved_name))


def write_summary(writer, global_step, avg_loss,max_loss,min_loss, saved_name,sess,log_summary_op, summary_text_tensor):
  eval_summary = tf.Summary()  # proto buffer
  eval_summary.value.add(node_name='avg_loss', simple_value=avg_loss, tag="train_eval/avg_loss")
  eval_summary.value.add(node_name='max_loss', simple_value=max_loss, tag="train_eval/max_loss")
  eval_summary.value.add(node_name='min_loss', simple_value=min_loss, tag="train_eval/min_loss")
  writer.add_summary(summary=eval_summary, global_step=global_step)

  # write log info to summary
  log_info = 'eval save model : global_step:{} avg_loss:{} \
              max_loss:{}  min_loss:{}  \n saved_file: {} '.format( \
      global_step, avg_loss, max_loss,min_loss,saved_name)
  log_summary=sess.run(fetches=[log_summary_op],
                       feed_dict={summary_text_tensor:log_info})
  writer.add_summary(summary=log_summary[0], global_step=global_step)
  writer.flush()


# TODO should use tf.natural_exp_decay

#TODO decay refer to distributed cifar-10.
def l_r_decay(initial_l_r, global_steps):
    return initial_l_r * math.exp(-0.3 * np.floor_divide(global_steps, TRAINING_CFG.l_r_decay_freq))



if __name__== '__main__':
  tf.logging.info("@@@  start grid domain-vin_tf training @@@ start time:{}".format(time.ctime()))
  tf.logging.set_verbosity(tf.logging.INFO)
  train()

