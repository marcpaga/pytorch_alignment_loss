"""
PyTorch implementation of the alignment loss used in DeepConsensus 0.3
https://github.com/google/deepconsensus/blob/3f7eca9c158d3b0448c6f6a4635dd27c8c601329/deepconsensus/models/losses_and_metrics.py

This is basically a Tensorflow to PyTorch translation, the invention of this 
loss function corresponds to the team that developed DeepConensus
https://github.com/google/deepconsensus
"""

from typing import Callable, Mapping, Optional, Tuple, Union

import torch
from torch import nn


class AlignmentLoss(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        del_cost: Optional[float] = 1.0,
        loss_reg: Optional[float] = 1.0,
        width: Optional[int] = None,
        reduction: Optional[str] = 'mean',
        pad_token: Optional[int] = 1,
        *args, 
        **kwargs
        ):
        super(AlignmentLoss, self).__init__(*args, **kwargs)

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError('reduction has to be "mean", "sum" or "none"')

        self.num_tokens = num_tokens
        self.del_cost = del_cost
        self.loss_reg = loss_reg
        self.width = width
        self.reduction = reduction
        self.pad_token = pad_token

    def forward(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        ) -> torch.Tensor:
        """Calculate the loss
        Args:
            y_true (torch.Tensor): tensor with the ground truth labels with
                shape [batch, targets_length]
            y_pred (torch.Tensor): tensor with the predictions with shape 
                [batch, predictions_length, n_tokens]. 
        Note:
            predictions_length >= targets_length

        Returns:
            A torch.Tensor with the value of the loss
        """

        dtype = y_pred.dtype
        inf = torch.Tensor([1e9], dtype = dtype)

        y_true, seq_lens = self.preprocess_y_true(y_true)
        y_pred = self.preprocess_y_pred(y_pred)

        subs_costs = self.xentropy_subs_cost_fn(y_true, y_pred)
        ins_costs = self.xentropy_ins_cost_fn(y_pred)
        del_cost = torch.Tensor([self.del_cost], dtype)

        if self.width is None:
            loss = self.alignment(subs_costs, ins_costs, del_cost, seq_lens, inf, dtype)
        else:
            loss = self.banded_alignment(subs_costs, ins_costs, del_cost, seq_lens, inf, dtype)
        
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def preprocess_y_true(
        self,
        y_true: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies AlignmentLoss-specific preprocessing to labels tensor.
        Args:
        y_true: A torch.Tensor [batch, m] representing the ground-truth
            sequences.
        dtype: The dtype for the one-hot encoded output tensor of sequence labels.
        Returns:
        A tuple (y_true_oh, seq_lens) such that
            +  y_true_oh is a torch.Tensor [batch, m, n_tokens], where n_tokens
            is the number of tokens. It contains a one-hot
            representation of the input y_true, with pad or gap tokens removed
            tokens removed and extra pad tokens appended if necessary.
            +  seq_lens is a torch.Tensor [batch] containing the length of each
            label sequence in y_true, excluding any pad and gap tokens.
        """
        # Ensures y_true is of integer type.
        y_true = y_true.to(int)
        # Removes internal gaps, shifting sequences left and adding padding when
        # necessary.
        y_true = self.left_shift_sequence(y_true, self.pad_token)
        # Computes per-example label sequence length, excluding padding.
        seq_lens = torch.sum((y_true != self.pad_token).to(y_true.dtype), dim = -1)
        # Converts y_true to one-hot.
        
        y_true_oh = torch.nn.functional.one_hot(y_true, num_classes=self.num_tokens).to(dtype)
        return y_true_oh, seq_lens

    @staticmethod
    def left_shift_sequence(
        seq: torch.Tensor,
        pad_token : int,
        ) -> torch.Tensor:
        """Removes internal gaps and shifts labels to the left.
        Args:
            seq: Label tensor.
        Returns:
            left shifted seq
        """
        
        shape = seq.shape
        seq_length = shape[1]

        ixs = torch.arange(start=0, end=seq_length, dtype=int).repeat(shape[0]).view(shape[0], -1)
        # Sorting is performed in 2 stages. Sort internal gaps back by increasing
        # an index by the seq length, perform sort, then subtract to return
        # original index.
        sort_order = torch.sort(torch.where(seq != pad_token, ixs, seq_length + ixs)).values
        sort_order = torch.where(sort_order < seq_length, sort_order,
                                sort_order - seq_length)
        seq_left_aligned = torch.gather(seq, dim=1, index=sort_order)
        return seq_left_aligned

    @staticmethod
    def preprocess_y_pred(y_pred: torch.Tensor) -> torch.Tensor:
        # Ensures predicted scores add to one.
        y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
        return y_pred

    def xentropy_subs_cost_fn(
        self,
        y_true: torch.Tensor, 
        y_pred: torch.Tensor, 
        eps: float = 1e-7,
        ) -> torch.Tensor:
        """Pointwise cross-entropy substitution cost function for alignment loss.
        Args:
            y_true (torch.Tensor): [batch, m, n_tokens] representing the one-hot
            encoded ground-truth sequences.
            y_pred (torch.Tensor): [batch, n, n_tokens] representing the scores for
            for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
            probability simplex.
            eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
            - eps] for numerical stability.
        Returns:
            A torch.Tensor [batch, m, n] such that out[b][l1][l2] represents the
            (sparse) cross-entropy loss between y_true[b][l1] and y_pred[b][l2].
        """

        y_pred = torch.clamp(y_pred, min = eps, max = 1 - eps)
        y_true, y_pred = torch.expand(y_true, 2), torch.expand(y_pred, 1)
        return -torch.sum(torch.xlogy(y_true, y_pred), dim =-1)

    def xentropy_ins_cost_fn(
        self,
        y_pred: torch.Tensor, 
        eps=1e-7) -> torch.Tensor:
        """Pointwise cross-entropy insertion cost function for alignment loss.
        Args:
            y_pred: A torch.Tensor [batch, n, n_tokens] representing the scores for
            for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
            probability simplex.
            eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
            - eps] for numerical stability.
        Returns:
            A torch.Tensor [batch, n] such that out[b][l] represents the
            cross-entropy loss between gap token and y_pred[b][l].
        """

        ins_scores = torch.clamp(y_pred[:, self.gap_token], min = eps, max = 1 - eps)
        return -torch.log(ins_scores)


