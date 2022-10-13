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
import numpy as np


class AlignmentLoss(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        del_cost: Optional[float] = 1.0,
        loss_reg: Optional[float] = None, #TODO change to 1.0
        width: Optional[int] = None,
        reduction: Optional[str] = 'sum',
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

        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        dtype = y_pred.dtype
        inf = torch.tensor([1e9], dtype = dtype)

        y_true, seq_lens = self.preprocess_y_true(y_true)
        y_pred = self.preprocess_y_pred(y_pred)

        subs_costs = self.xentropy_subs_cost_fn(y_true, y_pred)
        ins_costs = self.xentropy_ins_cost_fn(y_pred)
        del_cost = torch.tensor([self.del_cost], dtype = dtype)

        if self.width is None:
            loss = self.alignment(subs_costs, ins_costs, del_cost, seq_lens, inf, dtype)
        else:
            raise NotImplementedError
            loss = self.banded_alignment(subs_costs, ins_costs, del_cost, seq_lens, inf, dtype)
        
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

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
        y_true, y_pred = y_true.unsqueeze(2), y_pred.unsqueeze(1)
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

        ins_scores = torch.clamp(y_pred[..., self.pad_token], min = eps, max = 1 - eps)
        return -torch.log(ins_scores)

    def alignment(self, subs_costs, ins_costs, del_cost, seq_lens, inf, dtype):
        """Computes the alignment score values.
        Args:
            subs_costs: A tf.Tensor<float>[batch, len_1, len_2] input matrix of
            substitution costs.
            ins_costs: A tf.Tensor<float>[batch, len_1] input vector of insertion
            costs.
            del_cost: A float, the cost of deletion.
            seq_lens: A tf.Tensor<int>[batch] input matrix of true sequence lengths.
            inf: A float with very high value.
            dtype: the data type of y_pred
        Returns:
            A tf.Tensor<float>[batch] of values of the alignment scores.
        """
        # Gathers shape variables.
        shape = subs_costs.shape
        b, m, n = shape[0], shape[1], shape[2]
        # Computes and rearranges cost tensors for vectorized wavefront iterations.
        subs_costs = self.wavefrontify(subs_costs)
        ins_costs = self.wavefrontify_vec(ins_costs, m + 1)

        # TODO
        # Sets up reduction operators.
        if self.loss_reg is None:
            minop = lambda t: torch.min(t, 0)
        else:
            raise NotImplementedError()
            loss_reg = tf.convert_to_tensor(self.loss_reg, dtype)
            minop = lambda t: -loss_reg * tf.reduce_logsumexp(-t / loss_reg, 0)

        # Initializes recursion.
        v_opt = torch.full(size = (b, ), fill_value=inf.item())
        v_p2 = torch.concat([torch.zeros((1, b)), torch.full((m-1, b), inf.item())])
        v_p1 = torch.concat([
            ins_costs[0][:b-1],
            torch.full((1, b), del_cost.item()),
            torch.full((m-1, b), inf.item())
        ])

        i_range = torch.arange(m + 1, dtype=int)
        k_end = seq_lens + n  # Indexes antidiagonal containing last entry, w/o pad.
        # Indexes last entries in "wavefrontified" slices, accounting for padding.
        nd_indices = torch.stack([seq_lens, torch.arange(b, dtype=int)], -1)

        # Runs forward recursion.
        for k in torch.arange(2, m + n + 1):
            
            # Masks invalid entries in "wavefrontified" value tensor.
            j_range = k - i_range
            inv_mask = torch.logical_and(j_range >= 0, j_range <= n).unsqueeze(-1)
            
            o_m = v_p2 + subs_costs[k - 2]  # [m, b]
            o_i = v_p1 + ins_costs[k - 1]  # [m + 1, b]
            v_p2 = v_p1[:-1]
            o_d = v_p2 + del_cost  # [m, b]
            
            v_p1 = torch.concat(
                [o_i[:b-1][:],
                minop(torch.stack([o_m, o_i[1:], o_d]))[0]], 0)
            v_p1 = torch.where(inv_mask, v_p1, inf)
            v_opt = torch.where(k_end == k, v_p1[list(nd_indices.T)], v_opt)

        return v_opt

    def wavefrontify(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rearranges batch of input 2D tensors for vectorized wavefront algorithm.
        Args:
            tensor: A torch.Tensor [batch, len1, len2].
        Returns:
            A single torch.Tensor [len1 + len2 - 1, len1, batch] satisfying
            out[k][i][n] = t[n][i][k - i]
            if the RHS is well-defined, and 0 otherwise.
            In other words, for each len1 x len2 matrix t[n], out[..., n] is a
            (len1 + len2 - 1) x len1 matrix whose rows correspond to antidiagonals of
            t[n].
        """
        
        # b, l1, l2 = tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(tensor)[2]
        shape = tensor.shape
        b, l1, l2 = shape[0], shape[1], shape[2]
        # n_pad, padded_len = l1 - 1, l1 + l2 - 1
        n_pad, padded_len = l1 - 1, l1 + l2 - 1
        # ta = tf.TensorArray(tensor.dtype, size=l1, clear_after_read=True)
        ta = torch.zeros((l1, b, padded_len), dtype=tensor.dtype)
        for i in range(l1):
            row_i = tensor[:, i, :]
            row_i = torch.nn.functional.pad(row_i, [n_pad, n_pad])
            row_i = row_i[:, n_pad-i:-i+row_i.shape[1]]
            ta[i, ...] = row_i

        return ta.permute(2, 0, 1) # out[padded_len, l1, b]

    def wavefrontify_vec(self, tensor: torch.Tensor, len1: int) -> torch.Tensor:
        """Rearranges batch of 1D input tensors for vectorized wavefront algorithm.
        Args:
            tensor: A torch.Tensor[batch, len2].
            len1: An integer corresponding to the length of y_true plus one.
        Returns:
            A single torch.Tensor[len1 + len2 - 1, len1, batch] satisfying
            out[k][i][n] = t[n][k - i]
            if the RHS is well-defined, and 0 otherwise.
        """
        shape = tensor.shape
        b, len2 = shape[0], shape[1]
        n_pad, padded_len = len1 - 1, len1 + len2 - 1

        ta = torch.zeros((len1, b, padded_len))
        for i in range(len1):
            row_i = torch.nn.functional.pad(tensor, [n_pad, n_pad])
            row_i = row_i[:, n_pad-i:-i+row_i.shape[1]]
            ta[i, ...] = row_i
        return ta.permute(2, 0, 1) # out[padded_len, len1, b]