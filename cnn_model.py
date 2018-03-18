"""
implement a CNN network as mentioned in VIN paper.
Author: kenneth yu
"""

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, max_pool2d, dropout

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
    up
    left
    right
    down

=== state space:
    s_image of grid map
    s_goal
    s_curr_pos of current state
    
'''

TRAINING_CFG = tf.app.flags.FLAGS  # alias


class CNNModel:

    def __init__(self, cnn_model_cfg,optimizer,is_training, scope="cnn_model"):
        self.cnn_model_cfg = cnn_model_cfg
        self.optimizer = optimizer
        self.scope = scope
        self.is_training = is_training

    def create_net(self, state_inputs, labels, global_step_tensor):
        """
        :param labels:
        :param global_step_tensor:
        :param state_inputs:
        :return:
        """

        prev_layer = state_inputs
        conv_layers = []
        fc_layers = []

        with tf.variable_scope(self.scope):
            # conv layers
            # TODO add batch_norm to input process.
            for (n_maps, kernel_size, stride, padding, activation, initializer, normalizer, norm_param,
                 regularizer, pooling_kernel_size, pooling_stride, keep_prob) in \
                zip(
                self.cnn_model_cfg.conv_n_feature_maps, self.cnn_model_cfg.conv_kernel_sizes,
                self.cnn_model_cfg.conv_strides, self.cnn_model_cfg.conv_paddings, self.cnn_model_cfg.conv_activations,
                self.cnn_model_cfg.conv_initializers, self.cnn_model_cfg.conv_normalizers, self.cnn_model_cfg.conv_norm_params,
                self.cnn_model_cfg.conv_regularizers, self.cnn_model_cfg.pooling_kernel_sizes, self.cnn_model_cfg.pooling_strides,
                self.cnn_model_cfg.conv_dropout_keep_probs):
                prev_layer = conv2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size,
                                    stride=stride, padding=padding,
                                    activation_fn=activation,
                                    data_format='NHWC',
                                    normalizer_fn=normalizer,
                                    normalizer_params=norm_param,
                                    weights_initializer=initializer,
                                    weights_regularizer=regularizer,
                                    trainable=True)
                if pooling_kernel_size:
                    # max pooling only
                    prev_layer = max_pool2d(prev_layer, pooling_kernel_size, pooling_stride,
                                            padding='VALID', data_format='NHWC')
                if keep_prob < 1:
                    prev_layer = dropout(prev_layer, keep_prob,is_training=self.is_training)
                conv_layers.append(prev_layer)

            ##fc layers.inc output layer.
            # flatten the output of last conv layer to (batch_size, n_fc_in)
            prev_layer = tf.reshape(conv_layers[-1], shape=[-1,conv_layers[-1].shape[1] * conv_layers[-1].shape[2] * conv_layers[-1].shape[3]])
            for n_unit, activation, initializer, normalizer, norm_param, regularizer,keep_prob \
                    in zip(
                    self.cnn_model_cfg.n_fc_units, self.cnn_model_cfg.fc_activations, self.cnn_model_cfg.fc_initializers,
                    self.cnn_model_cfg.fc_normalizers, self.cnn_model_cfg.fc_norm_params, self.cnn_model_cfg.fc_regularizers,
                    self.cnn_model_cfg.fc_dropout_keep_probs):
                prev_layer = fully_connected(prev_layer, num_outputs=n_unit,
                                             activation_fn=activation,
                                             weights_initializer=initializer,
                                             normalizer_fn=normalizer,
                                             normalizer_params=norm_param,
                                             weights_regularizer=regularizer,
                                             trainable=True)
                if keep_prob < 1:
                    prev_layer = dropout(prev_layer, keep_prob, is_training=self.is_training)
                fc_layers.append(prev_layer)

            # logits should be [batch_size, num_action]
            logits = prev_layer

            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
            total_loss = tf.add_n(reg_loss + [cross_entropy_loss], name='total_loss')
            train_op = self.optimizer.minimize(total_loss, global_step_tensor)

            total_loss_mean = tf.reduce_mean(total_loss)
            with tf.name_scope('loss'):
                tf.summary.scalar(name='total_loss', tensor=total_loss_mean,
                                  collections=[TRAINING_CFG.summary_keys])

            # with tf.name_scope('d_policy_loss_da_grads'):
            #   d_policy_loss_da_grads=tf.gradients(ys=policy_loss,xs=actor.action_bounded_tensors)
            #   for i in range(len(dq_da_grads)):
            #     tf.summary.scalar(name='d_policy_loss_da_grads_'+str(i)+'norm',tensor=tf.norm(d_policy_loss_da_grads[i]),
            #                       collections=[self.cnn_model_cfg.actor_summary_keys])

        # == end with variable_scope() ==
        return train_op, total_loss
