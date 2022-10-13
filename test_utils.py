from typing import List, Text, Tuple, Union
import numpy as np


GAP_OR_PAD = ' '
ALLOWED_BASES = 'ATCG'
VOCAB = GAP_OR_PAD + ALLOWED_BASES
NP_DATA_TYPE = np.float32



def seq_to_array(seq: str) -> List[int]:
  return [VOCAB.index(i) for i in seq]

def get_one_hot(value: Union[int, np.ndarray]) -> np.ndarray:
  """Returns a one-hot vector for a given value."""
  return np.eye(len(VOCAB), dtype=NP_DATA_TYPE)[value]

def multiseq_to_array(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG sequences to DC numeric format."""
  return np.array(list(map(seq_to_array, sequences)))


def seq_to_one_hot(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG to one-hot format."""
  result = []
  for seq in sequences:
    result.append(get_one_hot(multiseq_to_array(seq)))
  result = np.squeeze(result)
  return result.astype(NP_DATA_TYPE)


def convert_seqs(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
  """Creates label and associated y_pred tensor.
  Args:
    sequences: string array inputs for label and prediction
  Returns:
    y_true as array
    y_pred_scores as probability array
  """
  y_true, y_pred_scores = sequences
  y_true = multiseq_to_array(y_true).astype(NP_DATA_TYPE)
  y_pred_scores = seq_to_one_hot(y_pred_scores)
  return y_true, y_pred_scores