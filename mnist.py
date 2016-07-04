"""
A simple 6 layer Convolutional Neural Network to recognize handwritten digits. 
On the MNIST dataset (http://yann.lecun.com/exdb/mnist/), this model achieves a 0.5% test error. 

This is meant as a tutorial/introduction to TensorFlow and is based on the MNIST model example in TensorFlow 
(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py). 
Read the accompanying README for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import getpass
import sys
import time
import gzip
import numpy
import tensorflow as tf
import argparse
from six.moves import xrange

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  IMAGE_SIZE = 28
  NUM_CHANNELS = 1
  PIXEL_DEPTH = 255
  NUM_LABELS = 10
  VALIDATION_SIZE = 5000
  SEED = 66478  # Set to None for random seed.
  BATCH_SIZE = 64
  NUM_EPOCHS = 20
  EVAL_BATCH_SIZE = 64
  EVAL_FREQUENCY = 100
  L2_REG = 5e-4
  BASE_LR = 0.01
  DROPOUT = 0.5

class DigitRecognizer():
  """
  A Convolutional Neural Network for digit recognition. The model architecture is:
  [conv1 - relu1 - pool1] - [conv2 - relu2 - pool2] - [conv3 - relu3] - dropout - fc1 - dropout - fc2
  """

  def extract_data(self, filename, num_images):
    """
    Extract the images into a tensor of the dimenstions: [num images, image width, image height, num channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    with gzip.open(filename) as bytestream:
      bytestream.read(16)
      buf = bytestream.read(self.config.IMAGE_SIZE * self.config.IMAGE_SIZE * num_images)
      data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
      data = (data - (self.config.PIXEL_DEPTH / 2.0)) / self.config.PIXEL_DEPTH
      data = data.reshape(num_images, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 1)
      return data

  def extract_labels(self, filename, num_images):
    """
    Extract the labels into a vector of label IDs.
    """
    with gzip.open(filename) as bytestream:
      bytestream.read(8)
      buf = bytestream.read(1 * num_images)
      labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels

  def error_rate(self, predictions, labels):
    """
    Return the error rate based on dense predictions and one-hot labels.
    """
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])

  def add_placeholders(self):
    """
    add placeholders variables. These placeholder nodes will be fed a batch of training data at each
    training step.
    """
    self.train_data_node = tf.placeholder(tf.float32, shape=(self.config.BATCH_SIZE, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, self.config.NUM_CHANNELS))
    self.train_labels_node = tf.placeholder(tf.int64, shape=(self.config.BATCH_SIZE,))
    self.eval_data = tf.placeholder(tf.float32, shape=(self.config.EVAL_BATCH_SIZE, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, self.config.NUM_CHANNELS))

  def create_feed_dict(self, input_data, input_labels=None):
    """Creates the feed_dict. A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    
    Args:
      input_data: A batch of input data.
      input_labels: A batch of label data.
    """
    feed_dict = {}
    if input_labels is not None:
      feed_dict[self.train_data_node] = input_data
      feed_dict[self.train_labels_node] = input_labels
    else:
      feed_dict[self.eval_data] = input_data

    return feed_dict

  def add_model_vars(self):
    """
    The variables below hold all the trainable weights. They are passed an
    initial value which will be assigned when we call tf.initialize_all_variables().run()
    """
    # conv1: 5x5 filter, depth 32.
    self.conv1_weights = tf.Variable(tf.truncated_normal([5, 5, self.config.NUM_CHANNELS, 32], stddev=0.1, seed=self.config.SEED))
    self.conv1_biases = tf.Variable(tf.zeros([32]))

    #conv2: 5x5 filter, depth 64
    self.conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=self.config.SEED))
    self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    #conv3: 5x5 filter, depth 64
    self.conv3_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1, seed=self.config.SEED))
    self.conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    #fc1: 512 hidden dims
    self.fc1_weights = tf.Variable(tf.truncated_normal([self.config.IMAGE_SIZE // 4 * self.config.IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=self.config.SEED))
    self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    #fc2
    self.fc2_weights = tf.Variable(tf.truncated_normal([512, self.config.NUM_LABELS], stddev=0.1, seed=self.config.SEED))
    self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.config.NUM_LABELS]))


  def add_model(self, data, train=False, return_hidden=False):
    """
    Construct the computational graph that defines our CNN model
    """
    #conv. layer 1
    conv1 = tf.nn.conv2d(data, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #conv. layer 2
    conv2 = tf.nn.conv2d(pool1, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #conv. layer 3 with dropout
    conv3 = tf.nn.conv2d(pool2, self.conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_biases))
    if train:
      relu3 = tf.nn.dropout(relu3, self.config.DROPOUT, seed=self.config.SEED)

    # Reshape the feature map cuboid to feed it to the fully connected layers.
    relu_shape = relu3.get_shape().as_list()
    reshaped = tf.reshape(
        relu3,
        [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])

    #fully connected layer 1 with dropout
    hidden = tf.nn.relu(tf.matmul(reshaped, self.fc1_weights) + self.fc1_biases)
    if train:
      hidden = tf.nn.dropout(hidden, self.config.DROPOUT, seed=self.config.SEED)

    #add regularization loss
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 
      tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.conv1_weights) + tf.nn.l2_loss(self.conv2_weights) + tf.nn.l2_loss(self.conv3_weights))

    #fully connected layer 2 (output layer)
    if return_hidden:
      return hidden
    else:
      return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

  def add_loss_op(self, logits):
    """Adds cross_entropy_loss ops to the computational graph.

    Args:
      logits: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    #data loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.train_labels_node))
    #reg. loss     
    loss += self.config.L2_REG * tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
    return loss


  def add_training_op(self, loss, train_size):
    """
    Sets up the training Ops.
    Creates an optimizer and applies the gradients to all trainable variables.
    """
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        self.config.BASE_LR,               
        batch * self.config.BATCH_SIZE,  
        train_size,          
        0.95,                
        staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=batch)
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    #config
    self.config = config
    #extract data
    self.train_data = self.extract_data('data/train-images-idx3-ubyte.gz', 60000)
    self.train_labels = self.extract_labels('data/train-labels-idx1-ubyte.gz', 60000)
    self.test_data = self.extract_data('data/t10k-images-idx3-ubyte.gz', 10000)
    self.test_labels = self.extract_labels('data/t10k-labels-idx1-ubyte.gz', 10000)

    self.add_placeholders()
    self.add_model_vars()
    logits = self.add_model(self.train_data_node, train=True)
    self.loss = self.add_loss_op(logits)
    self.train_op = self.add_training_op(self.loss, self.train_labels.shape[0])

    # Predictions for the current training minibatch.
    self.train_prediction = tf.nn.softmax(logits)
    # Predictions for the test and validation, which we'll compute less often.
    self.eval_prediction = tf.nn.softmax(self.add_model(self.eval_data, train=False))
    #feature extraction for t-SNE viz.
    self.featurize = self.add_model(self.eval_data, train=False, return_hidden=True)

  def eval_in_batches(self, data, session):
    """
    Get all predictions for a dataset by running it in small batches.
    """
    size = data.shape[0]
    eval_batch_size = self.config.EVAL_BATCH_SIZE
    if size < eval_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, self.config.NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, eval_batch_size):
      end = begin + eval_batch_size
      if end <= size:
        predictions[begin:end, :] = session.run(
            self.eval_prediction,
            feed_dict=self.create_feed_dict(data[begin:end, ...]))
      else:
        batch_predictions = session.run(
            self.eval_prediction,
            feed_dict=self.create_feed_dict(data[-eval_batch_size:, ...]))
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  def featurize_in_batches(self, session, sample_size):
    """
    Get fully connected layer1 features by running it in small batches.
    """
    batch_size = self.config.EVAL_BATCH_SIZE
    if sample_size < batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    features = numpy.ndarray(shape=(sample_size, 512), dtype=numpy.float32)
    labels = numpy.ndarray(shape=(sample_size,),dtype=numpy.int64)
    for begin in xrange(0, sample_size, batch_size):
      end = begin + batch_size
      if end <= sample_size:
        features[begin:end, :] = session.run(
            self.featurize,
            feed_dict=self.create_feed_dict(self.train_data[begin:end, ...]))
        labels[begin:end] = self.train_labels[begin:end]
      else:
        features[end-batch_size:end,:] = session.run(
            self.featurize,
            feed_dict=self.create_feed_dict(self.train_data[end-batch_size:end, ...]))
        labels[end-batch_size:end] = self.train_labels[end-batch_size:end]
    return features, labels
    

  def run_training(self, session):
    num_epochs = self.config.NUM_EPOCHS
    train_size = self.train_labels.shape[0]
    batch_size = self.config.BATCH_SIZE
    best_error = 2.0
    saver = tf.train.Saver()
    for step in xrange(int(num_epochs * train_size) // batch_size):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * batch_size) % (train_size - batch_size)
      batch_data = self.train_data[offset:(offset + batch_size), ...]
      batch_labels = self.train_labels[offset:(offset + batch_size)]

      #create feed dictionary
      feed_dict = self.create_feed_dict(batch_data, batch_labels)
      # Run the graph and fetch some of the nodes.
      _, loss, predictions = session.run(
          [self.train_op, self.loss, self.train_prediction],
          feed_dict=feed_dict)
      if step % self.config.EVAL_FREQUENCY == 0:
        print('Step %d (epoch %.2f)' %
              (step, float(step) * batch_size / train_size))
        print('Minibatch loss: %.3f' % (loss))
        train_error = self.error_rate(predictions, batch_labels)
        print('Minibatch error: %.1f%%' % train_error)
        validation_predictions = self.eval_in_batches(self.test_data, session)
        validation_error = self.error_rate(validation_predictions, self.test_labels)
        print('Test error: %.1f%%' % validation_error)
        if validation_error < best_error:
          saver.save(session, 'model_weights/weights')
          print("New best model saved")
          best_error = validation_error
        sys.stdout.flush()

def extract_features():
  """
  Extract features for t-SNE visualization
  """
  config = Config()
  config.EVAL_BATCH_SIZE = 50
  with tf.Graph().as_default():
    model = DigitRecognizer(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
      saver.restore(session, 'model_weights/weights')
      features,labels = model.featurize_in_batches(session, 10000)
      #save
      numpy.save('tsne_features.npy',features)
      numpy.save('tsne_labels.npy',labels)

def classify():
  """
  Test Digit Recognition model.
  """
  # Create a local session to run the training.
  config = Config()
  with tf.Graph().as_default():
    model = DigitRecognizer(config)
    init = tf.initialize_all_variables()
    with tf.Session() as session:
      print('Initializing...')
      session.run(init)
      print('Training...')
      model.run_training(session)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-features', action='store_true')
  parser.add_argument('-train', action='store_true')
  options = parser.parse_args()
  if options.train:
    classify()
  if options.features:
    extract_features()

    











