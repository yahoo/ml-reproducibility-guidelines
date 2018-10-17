// Copyright 2018 Oath Inc.
// Licensed under the terms of the MIT license. Please see LICENSE file in project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_NUM_CHANNELS = 3
_DEFAULT_IMAGE_SIZE = 224
_RESIZE_SIDE_MIN = 256


def _parse_function(example_proto, output_height, output_width, is_training=False,
                    resize_side_min=_RESIZE_SIDE_MIN):
    """Parse TF Records and preprocess image and labels
    """
    
    # parse the TFRecord
    feature_map = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/class/label": tf.VarLenFeature(tf.int64),
        "image/height": tf.FixedLenFeature((), tf.int64, default_value=0),
        "image/width": tf.FixedLenFeature((), tf.int64, default_value=0),
        "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
    }
    features = tf.parse_single_example(example_proto, feature_map)
    
    # parse bounding box
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    # ImageNet preprocessing 
    from official.resnet import imagenet_preprocessing
    image = imagenet_preprocessing.preprocess_image(
        image_buffer=features["image/encoded"],
        bbox=bbox,
        output_height=output_height,
        output_width=output_width,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    
    # convert labels to dense
    labels = tf.sparse_tensor_to_dense(features["image/class/label"])
    
    return image, labels


def train_parse_function(example_proto):
    return _parse_function(example_proto, 
                           _DEFAULT_IMAGE_SIZE, 
                           _DEFAULT_IMAGE_SIZE, 
                           is_training=True)


def eval_parse_function(example_proto):
    return _parse_function(example_proto, 
                           _DEFAULT_IMAGE_SIZE,
                           _DEFAULT_IMAGE_SIZE, 
                           is_training=False)


