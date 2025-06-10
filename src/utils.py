import numpy as np

def encode(message, G, p):
    """
    Encodes a message vector using generator matrix G over F_p.
    """
    message = np.array(message) % p
    G = np.array(G) % p
    return (message @ G) % p

def hamming_weight(vector):
    """
    Computes the Hamming weight of a vector (number of non-zero entries).

    Parameters:
        vector (array-like): Sequence of integers or booleans.

    Returns:
        int: Hamming weight (count of non-zero elements).
    """
    arr = np.asarray(vector)
    # Support boolean arrays as well
    return int(np.count_nonzero(arr))


def hamming_distance(v1, v2):
    """
    Computes the Hamming distance between two vectors of equal length.

    Parameters:
        v1, v2 (array-like): Sequences of same shape, contain integers or booleans.

    Returns:
        int: Number of positions at which the corresponding elements differ.

    Raises:
        ValueError: If v1 and v2 have different lengths.
    """
    arr1 = np.asarray(v1)
    arr2 = np.asarray(v2)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Hamming distance requires equal-length vectors, got {arr1.shape} vs {arr2.shape}")
    return int(np.count_nonzero(arr1 != arr2))

