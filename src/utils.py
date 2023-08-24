import numpy as np
import pandas as pd


def normalize(sequence: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, float, float]:
    # Normalize data
    s_min = min(sequence)
    s_max = max(sequence)
    sequence = (sequence - s_min) / (s_max - s_min)
    sequence = sequence.replace(np.nan, 0)

    return sequence, s_min, s_max


def denormalize(
    sequence: pd.DataFrame | np.ndarray, s_min: float, s_max: float
) -> pd.DataFrame:
    sequence = sequence * (s_max - s_min) + s_min

    return sequence


def split_sets(
    sequence: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(sequence) * train_ratio)
    train, test = sequence[:train_size], sequence[train_size:]

    return train, test


def split_sequence(sequence, n_steps):
    x, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_idx = i + n_steps
        # Check if we are beyond the sequence
        if end_idx > len(sequence) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


def split_multivariate_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
