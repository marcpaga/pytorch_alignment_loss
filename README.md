# pytorch_alignment_loss

PyTorch implementation of the alignment loss used in DeepConsensus 0.3
https://github.com/google/deepconsensus/blob/3f7eca9c158d3b0448c6f6a4635dd27c8c601329/deepconsensus/models/losses_and_metrics.py

This is basically a Tensorflow to PyTorch translation, the invention of this 
loss function corresponds to the team that developed DeepConensus
https://github.com/google/deepconsensus

The translation is incomplete, as only hard alignments have been implemented.

## How to use

Simply copy the contents of `loss.py` into your project and import the `AlignmentLoss` class.

### Args

num_tokens = len(vocab),
del_cost = 1.0, 
loss_reg = None, 
width = None, 
reduction = 'mean', 
pad_token 0,

### Demo

```
from loss import AlignmentLoss

# DNA vocab + gap for insertions/deletions
vocab = [' ', 'A', 'C', 'G', 'T']

# example true labels input
y_true = torch.tensor(
    [[2., 2., 1., 4., 4., 3.],
    [1., 2., 3., 4., 1., 3.],
    [1., 4., 3., 2., 4., 4.]], 
    dtype = torch.float32
) # shape = [batch, true_length]

# example output scores from the model

y_pred = torch.tensor(
    [[[0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0.]],

    [[0., 1., 0., 0., 0.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0.]],

    [[0., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 1.]]],
    dtype=torch.float32
) #shape = [batch, pred_length, num_tokens]

# note that pred_length >= true_length

align_loss = AlignmentLoss(
    num_tokens = len(vocab),
    del_cost = 1.0, 
    loss_reg = None, 
    width = None, 
    reduction = 'mean', 
    pad_token 0,
)
loss = align_loss(y_true, y_pred)

loss # tensor(16.1181)

```