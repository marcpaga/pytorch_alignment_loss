import unittest
from absl.testing import absltest
from absl.testing import parameterized

import torch
from loss import AlignmentLoss
from loss_tf import AlignmentLossTF
from test_utils import convert_seqs, VOCAB


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


class AlignmentLossTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Hard, identical sequences, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with same pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
                                                    'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with different pad',
          sequences=(['TTAGGCAT', 'AGCTGG  '], ['TTAGGCAT  ', 'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['T TA G G C', 'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, with pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
                                                    'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost two, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TAGGC ', 'AGCGG ']),
          del_cost=2.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, two deletions at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAG  ', 'GCGG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, one error, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['ATAGGC', 'TGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, two errors, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one erroneous insertion, no pad',
          sequences=(['TTAGGC', 'ATCGAC',
                      'AGCTGG'], ['TTAGGCA', 'ATCCGAC', 'CAGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one deletion, small deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion, large deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1e9,
          loss_reg=None,
          expected_loss=64.472,  # 4*log(eps), with eps = 1e-7
          width=None),
      # TODO: included test cases for soft alignment.
    #   dict(
    #       testcase_name='with band, identical sequences',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band, one deletion at cost one, with pad',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=1.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band, identical sequences, with same pad',
    #       sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
    #                                                 'AGCTGG    ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=1),
    #   dict(
    #       testcase_name='with band, correct insertions only, no pad',
    #       sequences=(['TTAGGC   ', 'AGCTG   G'], ['T TAG G C', 'AGC   TGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=8),
    #   dict(
    #       testcase_name='with band, correct insertions only, with pad',
    #       sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
    #                                                 'AGC    TGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=8),
    #   dict(
    #       testcase_name='with band, two errors, no pad',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
    #       width=4),
    #   dict(
    #       testcase_name='with band of 2, two dels, one align, two pads',
    #       sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=2.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band of 1,one del, one align, two pads, one del',
    #       sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=18.118,  # 2.0 + log(eps), with eps = 1e-7
    #       width=1),
  )
  def test_alignment_loss(self, sequences, del_cost, loss_reg, width,
                          expected_loss):
    """Checks that edit distance calculation matches expected value."""
    y_true, y_pred_scores = convert_seqs(sequences)
    loss_obj = AlignmentLoss(
        num_tokens=len(VOCAB),
        del_cost=del_cost, 
        loss_reg=loss_reg, 
        width=width
    )
    loss = loss_obj(y_true, y_pred_scores)
    self.assertAlmostEqual(float(loss), expected_loss, places=2)


class AlignmentLossTestTF(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Hard, identical sequences, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with same pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
                                                    'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with different pad',
          sequences=(['TTAGGCAT', 'AGCTGG  '], ['TTAGGCAT  ', 'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['T TA G G C', 'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, with pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
                                                    'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost two, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TAGGC ', 'AGCGG ']),
          del_cost=2.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, two deletions at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAG  ', 'GCGG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, one error, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['ATAGGC', 'TGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, two errors, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one erroneous insertion, no pad',
          sequences=(['TTAGGC', 'ATCGAC',
                      'AGCTGG'], ['TTAGGCA', 'ATCCGAC', 'CAGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one deletion, small deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion, large deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1e9,
          loss_reg=None,
          expected_loss=64.472,  # 4*log(eps), with eps = 1e-7
          width=None),
      # TODO: included test cases for soft alignment.
    #   dict(
    #       testcase_name='with band, identical sequences',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band, one deletion at cost one, with pad',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=1.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band, identical sequences, with same pad',
    #       sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
    #                                                 'AGCTGG    ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=1),
    #   dict(
    #       testcase_name='with band, correct insertions only, no pad',
    #       sequences=(['TTAGGC   ', 'AGCTG   G'], ['T TAG G C', 'AGC   TGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=8),
    #   dict(
    #       testcase_name='with band, correct insertions only, with pad',
    #       sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
    #                                                 'AGC    TGG']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=0.0,
    #       width=8),
    #   dict(
    #       testcase_name='with band, two errors, no pad',
    #       sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
    #       width=4),
    #   dict(
    #       testcase_name='with band of 2, two dels, one align, two pads',
    #       sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=2.0,
    #       width=2),
    #   dict(
    #       testcase_name='with band of 1,one del, one align, two pads, one del',
    #       sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
    #       del_cost=1.0,
    #       loss_reg=None,
    #       expected_loss=18.118,  # 2.0 + log(eps), with eps = 1e-7
    #       width=1),
  )
  def test_alignment_loss(self, sequences, del_cost, loss_reg, width,
                          expected_loss):
    """Checks that edit distance calculation matches expected value."""
    y_true, y_pred_scores = convert_seqs(sequences)
    loss_obj = AlignmentLossTF(
        del_cost=del_cost, 
        loss_reg=loss_reg, 
        width=width
    )
    loss = loss_obj(y_true, y_pred_scores)
    self.assertAlmostEqual(float(loss), expected_loss, places=2)


if __name__ == '__main__':
    unittest.main()
    absltest.main()