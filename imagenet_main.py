# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import logging
from Quantize import fg,flr,fgBN,fBits
#from Quantize import layer_output

#from tensorpack.tfutils.varreplace import remap_variables

import tensorflow as tf
import numpy as np

import resnet_model
import inception_preprocessing


g_scale=128

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/data/dataset/imagenet_dataset/TFRecord256/',
    help='The directory where the ImageNet input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default='./model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=18, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_epochs', type=int, default=100,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

#_MOMENTUM = 0.9


_MOMENTUM = 1./4

_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500


def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(1024)]
        #for i in range(2)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def record_parser(value, is_training):
  """Parse an ImageNet record from `value`."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = inception_preprocessing.preprocess_image(
      image=image,
      height=_DEFAULT_IMAGE_SIZE,
      width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                        num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)
  
  network = resnet_model.imagenet_resnet_v2(
      params['resnet_size'], _LABEL_CLASSES, params['data_format'])
  
  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  
  print('<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>')
  print(logits.name)
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
  
  #cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
  
  #cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      #logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'BatchNorm' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 256, the learning rate should be 0.1.
    #initial_learning_rate = 0.1 * params['batch_size'] / 256
    initial_learning_rate = 0.05
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        #int(batches_per_epoch * epoch) for epoch in [20, 30, 40, 50]]
    values = [
        initial_learning_rate * decay for decay in [1, 0.12, 0.06, 0.03, 0.03]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    
    g_values = [128.,128.,32.,8.,2.]
    g_scale = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, g_values)
    tf.identity(g_scale, name='g_scale')
    
    
    learning_rate=flr(learning_rate)
    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gradTrainBatch = optimizer.compute_gradients(loss)
    
    
    
    grad=[]
    var=[]
    for grad_and_vars in gradTrainBatch:
        grad.append(grad_and_vars[0])
        var.append(grad_and_vars[1])
        
    
    def QuantizeG(gradTrainBatch):
        grads = []
        for grad_and_vars in gradTrainBatch:
            if grad_and_vars[1].name == 'conv2d/kernel:0' or   grad_and_vars[1].name.find('dense')>-1:
                
                grads.append([grad_and_vars[0]*1.0 , grad_and_vars[1] ])            
            elif   grad_and_vars[1].name.find('BatchNorm')>-1:
                
                grads.append([fgBN(grad_and_vars[0],1.0) , grad_and_vars[1] ])
                                
            else:                
                grads.append([fg(grad_and_vars[0],1.0,g_scale) , grad_and_vars[1] ])
                
        return grads
    
    gradTrainBatch=QuantizeG(gradTrainBatch)   
    

    
    Mom_Q=[]    
    Mom_W=[]
    
    w_vars=tf.trainable_variables()
    for w_var in w_vars:
        if w_var.name==('conv2d/kernel:0')  or   w_var.name.find('dense')>-1:
            Mom_W.append(tf.assign(w_var,w_var))
            print(w_var.name)
            print('**************************')
        
            
        else:
            Mom_W.append(tf.assign(w_var,fBits(w_var,24)))
    
    
    
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(gradTrainBatch, global_step=global_step)
      opt_slot_name=optimizer.get_slot_names()
      train_vars=tf.trainable_variables()
      for train_var in train_vars:
         mom_var=optimizer.get_slot(train_var,opt_slot_name[0])         
         if train_var.name == ('conv2d/kernel:0')  or   train_var.name.find('dense')>-1:
             print(mom_var.name)
         else:
             Mom_Q.append(tf.assign(mom_var,fBits(mom_var,13)))
      
      train_op=tf.group([train_op,Mom_Q,Mom_W])
    
      
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  accuracy5 = tf.metrics.mean(tf.nn.in_top_k(logits,tf.argmax(labels, axis=1),k=5))
      
  metrics = {'accuracy': accuracy, 'accuracy5':accuracy5}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  #os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  # Set up a RunConfig to only save checkpoints once per training cycle.

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.99
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False

  
  run_config = tf.estimator.RunConfig(session_config=config).replace(save_checkpoints_steps=10010)
  resnet_classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })
  

  eval_record=[]

  for train_epoch in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy',
        'g_scale':'g_scale',
        'probs':'final_dense:0'
        
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    
    
    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)


    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])
    
    
    step = resnet_classifier.get_variable_value('global_step')
    
    
    
    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)
    eval_record.append(eval_results)
    np.save('./data/eval_results.npy',eval_record)  
  np.save('./data/eval_results_final.npy',eval_record)



if __name__ == '__main__':
  #tf.logging.set_verbosity(tf.logging.INFO)
  cur_path = os.getcwd()
  print(cur_path)
  if not os.path.exists(cur_path+'/log'):
    os.mkdir(cur_path+'/log')
  if not os.path.exists(cur_path+'/data'):
    os.mkdir(cur_path+'/data')

  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)
  fh=logging.FileHandler(cur_path+'/log/tensorflow.log')
  fh.setLevel(logging.DEBUG)
  log.addHandler(fh)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
