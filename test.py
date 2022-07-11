import unittest
import torch
from loss import AlignmentLoss


class TestLeftShiftSequence(unittest.TestCase):

    def test_left_shift(self):
        seq_gap = torch.Tensor([
            [0, 3, 4, 0, 0, 1, 2],
            [3, 1, 0, 2, 0, 0, 1],
            [1, 2, 3, 4, 0, 0, 0],
        ])
        seq = torch.Tensor([
            [3, 4, 1, 2, 0, 0, 0],
            [3, 1, 2, 1, 0, 0, 0],
            [1, 2, 3, 4, 0, 0, 0],
        ])
        self.assertTrue(torch.equal(
            seq, 
            AlignmentLoss.left_shift_sequence(seq_gap, 0)
        ))

if __name__ == '__main__':
    unittest.main()