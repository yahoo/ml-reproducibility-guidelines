from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """
    """
    from official.resnet import resnet_model
    
    tf.summary.image('images', features, max_outputs=6)
    
    features = tf.cast(features, tf.float32)
    
    model = resnet_model.Model(
        resnet_size=50,
        bottleneck=True,
        num_classes=1000,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=[3, 4, 6, 3],
        block_strides=[1, 2, 2, 2],
        final_size=2048,
        resnet_version=2,
        data_format=None,
        dtype=tf.float32)
    
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)
    
    predictions = {
        'classes': tf.argmax(logits, axis=1) + 1,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })
    
    # calculate loss
    onehot_labels = tf.one_hot(tf.squeeze(labels - 1), 1000)
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    loss_filter_fn = exclude_batch_norm
    
    # add weight decay
    l2_loss = params['weight_decay'] * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() 
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    
    loss = cross_entropy + l2_loss
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        global_step = tf.train.get_or_create_global_step()
        
        learning_rate = tf.train.piecewise_constant(global_step, 
            params['lr_boundaries'], params['lr_values'])
        
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9)
        
        minimize_op = optimizer.minimize(loss, global_step)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        
        eval_metric_ops = None
        
    else:
        
        train_op = None
        
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
        
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        
        eval_metric_ops = {'accuracy': accuracy}
    
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)