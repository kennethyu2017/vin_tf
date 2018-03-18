"""
implement a CNN network as mentioned in VIN paper.
Author: kenneth yu
"""

import tensorflow as tf
from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt

# NUM_CLASSES = cnn_model_cfg.action_dims
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

TRAINING_CFG = tf.app.flags.FLAGS


class GridDomainReader(object):
    def __init__(self, train_filename, test_filename, im_size, batch_size, random_show=False):
        self.x_train, self.y_train, self.x_validation, self.y_validation, self.x_test, self.y_test = \
            self.read_grid_domains(train_filename, test_filename, im_size)
        self.train_sample_idx = 0
        self.validation_sample_idx = 0
        self.batch_size = batch_size

        if random_show:
            self.show_train_validation()
        return

    def num_examples_per_epoch(self,subset='train'):
        if subset == 'train':
            return self.x_train.shape[0]
        elif subset == 'validation':
            return self.x_validation.shape[0]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def read_grid_domains(self, train_filename, test_filename, im_size):
        # run training from input matlab data file, and save test data prediction in output file
        # load data from Matlab file, including
        # im_data: flattened images
        # value_data: flattened reward image
        # state_data: flattened state images
        # label_data: action idx value(0-7). not one-hot vector for action (state difference)
        #TODO split mat files and use input queues.
        matlab_data = sio.loadmat(train_filename)  # from matlab uint8 dtype.

        im_data = matlab_data["im_data"][0:1000]
        # TODO: normalize will kill precise position information?
        # im_data = (im_data - 1) / 255  # obstacles = 1, free zone = 0
        im_data = 1 - im_data  # make obstacles = 1, free zone = 0  # in matlab 0:obstacle and border,black color. 1: free space,white color.

        value_data = matlab_data["value_data"][:1000]  # goal is 10, other is 0.
        # state1_data = matlab_data["state_x_data"]  # flattened x pos
        # state2_data = matlab_data["state_y_data"][:1000]  # flattened y pos
        state_xy_data = matlab_data['state_xy_data'][:1000] # xy coordinate of each sample. start from 0.
        # label_data is idx action.
        y_data = matlab_data["label_data"][:1000]
        y_data=y_data.squeeze()

        del matlab_data

        # reshape img data from flatten to [num_samples, h, w]
        # TODO reshape and cast in tf.
        im_data = im_data.reshape((-1, im_size[0], im_size[1]))
        # reshape img value(prior,: 10@goal and -1@others) from flatten to [num_samples, h, w]
        value_data = value_data.reshape((-1, im_size[0], im_size[1]))

        state_data = np.zeros_like(im_data,'int8')
        for i in range(state_data.shape[0]):  # num_samples
            state_data[i, state_xy_data[i][0], state_xy_data[i][1]] = 1  # set position as value 1 for each sample
        state_data = state_data.astype('int8')
        # stack img + val + state(curr pos) along axis 1.
        # x_data shape [num_samples, h, w, 3] to comply with conv2d NHWC format.
        x_data = np.stack((im_data, value_data, state_data), axis=-1)

        del im_data,value_data,state_data

        # all_training_samples = int(6 / 7.0 * x_data.shape[0])
        num_train = int(6/7.0 * x_data.shape[0])
        # x_test = x_data[num_train + num_validation:]
        # y_test = y_data[num_train + num_validation:]

        # shuffle training and validation data
        shuffle_idx = np.random.permutation(x_data.shape[0])
        x_data = x_data[shuffle_idx]
        y_data = y_data[shuffle_idx]

        x_train = x_data[0:num_train]
        y_train = y_data[0:num_train]
        x_validation = x_data[num_train:]
        y_validation = y_data[num_train:]

        del x_data,y_data

        # TODO read test data.
        x_test = None
        y_test = None
        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def batch_train_inputs(self):
        x_batch = self.x_train[self.train_sample_idx:self.train_sample_idx + self.batch_size]
        y_batch = self.y_train[self.train_sample_idx:self.train_sample_idx + self.batch_size]
        self.train_sample_idx = (self.train_sample_idx + self.batch_size) % self.x_train.shape[0]
        return x_batch, y_batch

    def batch_validation_inputs(self):
        x_batch = self.x_validation[self.validation_sample_idx:self.validation_sample_idx + self.batch_size]
        y_batch = self.y_validation[self.validation_sample_idx:self.validation_sample_idx + self.batch_size]
        self.validation_sample_idx = (self.validation_sample_idx + self.batch_size) % self.x_validation.shape[0]
        return x_batch, y_batch

    # TODO test data
    def test_input(self):
        pass

    def sample_data_and_show(self,datas, labels,title):
        sample_idx = np.random.randint(0, datas.shape[0])
        [im, value, state] = np.split(datas[sample_idx], 3, axis=-1)

        im = np.squeeze(im)
        value = np.squeeze(value)
        state = np.squeeze(state)

        goal_rc = np.argwhere(value==10)[0] # row, col of goal.
        goal_y, goal_x = goal_rc[0],goal_rc[1] # coordinate of matplot, y->ndarray row, x->ndarray col.
                                              #NOTE: in matlab, x->row, y->col.
        state_rc = np.argwhere(state==1)[0] # row, col of curr pos.
        state_y, state_x = state_rc[0], state_rc[1]

        action_label = labels[sample_idx]
        action_rc_diff = [
            [-1, 0],  #N
            [1, 0],   #S
            [0, 1],
            [0, -1],
            [-1, 1],  #NE
            [-1, -1], #NW
            [1, 1],  #SE
            [1, -1]]  #SW
        next_state_rc = action_rc_diff[action_label] + state_rc
        next_state_y, next_state_x = next_state_rc[0],next_state_rc[1]

        plt.figure(1)
        plt.imshow(im,cmap=plt.cm.gray)
        plt.plot(goal_x,goal_y,'-go')
        plt.plot(state_x, state_y, '-rx')
        plt.plot(next_state_x,next_state_y, '-bs')
        plt.show(1)

    def show_train_validation(self):
        # TODO print img and traj to show
        self.sample_data_and_show(self.x_train, self.y_train,'train sample')
        self.sample_data_and_show(self.x_validation, self.y_validation, 'validation sample')


#
# def read_grid_domain_files(filename_queue):
#
#   pass
#
# def eval_image(image, height, width, scope=None):
#   """Prepare one image for evaluation.
#   """
#
# def distort_image(image, height, width):
#   # Image processing for training the network. Note the many random
#   # distortions applied to the image.
#
#   # TODO:Randomly flip the image horizontally, vertically,as augument of data.
#   # distorted_image = tf.image.random_flip_left_right(image)
#
#   #TODO: Set the shapes of tensors.
#   # image.set_shape([height, width, 3])
#   return image
#
#
# def image_preprocessing(image, train, thread_id=0):
#   """Decode and preprocess one image for evaluation or training.
#
#   Args:
#     image:  Tensor
#     train: boolean
#     thread_id: integer indicating preprocessing thread
#
#   Returns:
#     3-D float Tensor containing an appropriately scaled image
#
#   Raises:
#     ValueError: if user does not provide bounding box
#   """
#   height = TRAINING_CFG.im_size[0]
#   width = TRAINING_CFG.im_size[1]
#
#   if train:
#     image = distort_image(image, height, width, thread_id)
#   else:
#     image = eval_image(image, height, width)
#
#   # Finally, rescale to [-1,1] instead of [0, 1)
#   # image = tf.subtract(image, 0.5)
#   # image = tf.multiply(image, 2.0)
#   return image
#
#
# def batch_inputs(filename,im_size, batch_size, train):
#   """Contruct batches of training or evaluation examples from the image dataset.
#
#   Args:
#     data_dir:
#     batch_size: integer
#     train: boolean
#     num_preprocess_threads: integer, total number of preprocessing threads
#     num_readers: integer, number of parallel readers
#
#   Returns:
#     images: 4-D float Tensor of a batch of images
#     labels: 1-D integer Tensor of [batch_size].
#
#   Raises:
#     ValueError: if data is not found
#   """
#   with tf.name_scope('batch_processing'):
#     if not tf.gfile.Exists(filename):
#       raise ValueError('Failed to find file: ' + f)
#
#     # TODO: support multiple grid world files.
#     (x_train,y_train, x_validation,y_validation, x_test, y_test) = read_grid_domains(filename,im_size)
#
#     # TODO: to do image_proprocessing. augumenting imgs.
#     # image = image_preprocessing(image, train, thread_id)
#
#     # Display the training images in the visualizer.
#     # tf.summary.image('images', image_batch)
#
#     return image_batch, label_index_batch
#
#
# def distorted_inputs(batch_size=None, num_preprocess_threads=None,num_readers=None):
#   """
#   Distorting images provides a useful technique for augmenting the data
#   set during training in order to make the network invariant to aspects
#   of the image that do not effect the label.
#
#   Args:
#     batch_size: integer, number of examples in batch
#     num_preprocess_threads: integer, total number of preprocessing threads but
#       None defaults to FLAGS.num_preprocess_threads.
#
#   Returns:
#     images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
#                                        FLAGS.image_size, 3].
#     labels: 1-D integer Tensor of [batch_size].
#   """
#   if not batch_size:
#     batch_size = TRAINING_CFG.batch_size
#
#   # Force all input processing onto CPU in order to reserve the GPU for
#   # the forward inference and back-propagation.
#   with tf.device('/cpu:0'):  # tf.device can merge with /job:worker/task: #
#     image_batch, label_batch = batch_inputs(
#       file_name, batch_size, train=True)
#   return image_batch, label_batch


if __name__ == '__main__':
    reader = GridDomainReader('./data/gridworld_28.mat','./data/gridworld_28_test.mat',
                              [28,28],128)
    reader.show_train_validation()

