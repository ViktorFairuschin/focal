# Focal Loss

This is a Tensorflow implementation of the Focal Crossentropy Loss described in 
[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002).

## Usage

For imbalanced binary classification tasks:

```python
from focal import FocalBinaryCrossentropy


loss = FocalBinaryCrossentropy()

y_true = [0, 1, 0, 0]
y_pred = [0.3, 0.5, 0.7, 0.1]

loss(y_true, y_pred)
```

For imbalanced classification with two or more label classes:

```python
from focal import FocalCategoricalCrossentropy


loss = FocalCategoricalCrossentropy()

y_true = [[1., 0.], [0., 1.]]
y_pred = [[0.90, 0.1], [0.8, 0.2]]

loss(y_true, y_pred)
```