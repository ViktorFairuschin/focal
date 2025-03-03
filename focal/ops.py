# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf


EPSILON = 1.0e-9


def binary_crossentropy(targets: tf.Tensor, outputs: tf.Tensor, from_logits: bool) -> tf.Tensor:
    """ Computes binary crossentropy. """
    # if from_logits:
    #     return tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=outputs)

    outputs = tf.clip_by_value(outputs, EPSILON, (1.0 - EPSILON))

    bce = targets * tf.math.log(outputs + EPSILON)
    bce += (1 - targets) * tf.math.log(1 - outputs + EPSILON)
    return - bce


def focal_binary_crossentropy(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        alpha: float,
        gamma: float,
        from_logits: bool
) -> tf.Tensor:
    """ Computes focal binary crossentropy. """
    outputs = tf.convert_to_tensor(y_pred)
    targets = tf.cast(y_true, outputs.dtype)

    outputs = tf.math.sigmoid(outputs) if from_logits else outputs

    p_t = targets * outputs + (1 - targets) * (1 - outputs)

    # compute focal factor
    focal_factor = tf.pow(1.0 - p_t, gamma)

    # compute binary cross entropy
    bce = binary_crossentropy(targets, outputs, from_logits)

    focal_bce = focal_factor * bce

    # apply class balancing
    weight = targets * alpha + (1 - targets) * (1 - alpha)
    focal_bce = weight * focal_bce

    return focal_bce


def focal_categorical_crossentropy(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        alpha: float,
        gamma: float,
        from_logits: bool
) -> tf.Tensor:
    """ Computes focal categorical crossentropy. """
    outputs = tf.convert_to_tensor(y_pred)
    targets = tf.cast(y_true, outputs.dtype)

    outputs = tf.math.softmax(outputs) if from_logits else outputs
    outputs = outputs / tf.reduce_sum(outputs, axis=-1, keepdims=True)

    outputs = tf.clip_by_value(outputs, EPSILON, (1.0 - EPSILON))

    # compute categorical cross entropy
    cce = - targets * tf.math.log(outputs)

    # compute weighting factor
    weighting_factor = alpha * tf.pow(1.0 - outputs, gamma)

    focal_cce = weighting_factor * cce
    focal_cce = tf.reduce_sum(focal_cce, axis=-1)

    return focal_cce

