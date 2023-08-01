import numpy as np
import pandas as pd


def normalize(sequence: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    # Normalize data
    s_min = min(sequence)
    s_max = max(sequence)
    sequence = (sequence - s_min) / (s_max - s_min)

    return sequence, s_min, s_max


def split_sets(
    sequence: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(sequence) * train_ratio)
    train, test = sequence[:train_size], sequence[train_size:]

    return train, test


def split_sequence(sequence, n_steps):
    # Split sequence into samples
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
