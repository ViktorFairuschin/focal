# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper

from focal.ops import focal_binary_crossentropy, focal_categorical_crossentropy


class FocalBinaryCrossentropy(LossFunctionWrapper):
    """
    Computes the focal binary crossentropy loss
    as described in https://arxiv.org/abs/1708.02002.

    :param alpha: Weight balancing factor for positive class.
    :param gamma: A focusing parameter used to compute the focal factor.
    :param from_logits: Whether to interpret y_pred as a tensor of logit values.
    :param reduction: Type of reduction to apply to loss.
    :param name: Optional name for the instance.

    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            from_logits: bool = False,
            reduction=tf.keras.losses.Reduction.AUTO,
            name="focal_binary_crossentropy"
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

        super().__init__(
            fn=focal_binary_crossentropy,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            reduction=reduction,
            name=name
        )

    def get_config(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
            "reduction": self.reduction,
            "name": self.name,
        }


class FocalCategoricalCrossentropy(LossFunctionWrapper):
    """
    Computes the focal categorical crossentropy loss
    as described in https://arxiv.org/abs/1708.02002.

    :param alpha: Weight balancing factor for positive class.
    :param gamma: A focusing parameter used to compute the focal factor.
    :param from_logits: Whether to interpret y_pred as a tensor of logit values.
    :param reduction: Type of reduction to apply to loss.
    :param name: Optional name for the instance.

    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            from_logits: bool = False,
            reduction=tf.keras.losses.Reduction.AUTO,
            name="focal_categorical_crossentropy"
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

        super().__init__(
            fn=focal_categorical_crossentropy,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            reduction=reduction,
            name=name
        )

    def get_config(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
            "reduction": self.reduction,
            "name": self.name,
        }

