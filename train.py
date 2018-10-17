// Copyright 2018 Oath Inc.
// Licensed under the terms of the MIT license. Please see LICENSE file in project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from glob import glob

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('dataset_directory', '/tmp/',
                           'Dataset directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
tf.app.flags.DEFINE_string('models_repository', '/tmp',
                           'Path to tensorflow/models repository')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of GPU for distributed training')
tf.app.flags.DEFINE_integer('batch_size', 256,
                            'Total batch size')

FLAGS = tf.app.flags.FLAGS


from model import model_fn


def dataset_input_fn(filenames, batch_size, is_training):
    """
    """
    from preprocess import train_parse_function, eval_parse_function
    
    dataset = tf.data.TFRecordDataset(filenames) 
    parse_function = train_parse_function if is_training else eval_parse_function
    dataset = dataset.map(parse_function, num_parallel_calls=32)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(256)
    if is_training is True:
        dataset = dataset.repeat()
    
    return dataset 
    


def main(argv):
    """
    """
    
    # add path to tensorflow/models repository
    sys.path.append(FLAGS.models_repository)
    
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
    
    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy)
    
    # learning rate schedule
    lr_boundaries = [500000, 1000000]
    lr_values = [0.01, 0.001, 0.0001]

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_directory,
        params={'weight_decay': 1e-4,
                'lr_boundaries': lr_boundaries,
                'lr_values': lr_values},
        config=run_config)
    
    batch_size_per_gpu = FLAGS.batch_size // FLAGS.num_gpus
    
    def train_input_fn():
        train_filenames = glob(os.path.join(FLAGS.dataset_directory, "train*"))
        return dataset_input_fn(train_filenames, batch_size_per_gpu, True)
    
    def eval_input_fn():
        eval_filenames = glob(os.path.join(FLAGS.dataset_directory, "validation*"))
        return dataset_input_fn(eval_filenames, 50, False)
    
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2000000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=2000,
                                      start_delay_secs=600,
                                      throttle_secs=1800)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
    tf.app.run()