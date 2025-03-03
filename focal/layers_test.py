# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
from focal.layers import FocalBinaryCrossentropy, FocalCategoricalCrossentropy


class FocalBinaryCrossentropyTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()

    def testOutput(self):
        # batch size 1
        y_true = [0, 1, 0, 0]
        y_pred = [-18.6, 0.51, 2.94, -12.8]

        loss = FocalBinaryCrossentropy(from_logits=True)
        self.assertAlmostEqual(loss(y_true, y_pred).numpy(), 0.51, places=2)

        # batch size 2
        y_true = [[0, 1], [0, 0]]
        y_pred = [[-18.6, 0.51], [2.94, -12.8]]

        loss = FocalBinaryCrossentropy(gamma=3, from_logits=True)
        self.assertAlmostEqual(loss(y_true, y_pred).numpy(), 0.482, places=3)

        # sample weight
        self.assertAlmostEqual(loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy(), 0.097, places=3)


class FocalCategoricalCrossentropyTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()

    def testOutput(self):
        y_true = [[0., 1., 0.], [0., 0., 1.]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        loss = FocalCategoricalCrossentropy()
        self.assertAlmostEqual(loss(y_true, y_pred).numpy(), 0.233, places=3)

        # sample weight
        self.assertAlmostEqual(loss(y_true, y_pred, sample_weight=[0.3, 0.7]).numpy(), 0.163, places=3)

        # different reduction
        loss = FocalCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.assertAlmostEqual(loss(y_true, y_pred).numpy(), 0.466, places=3)


if __name__ == '__main__':
    tf.test.main()

