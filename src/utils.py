import numpy as np

def encode(message, G, p):
    """
    Encodes a message vector using generator matrix G over F_p.
    """
    message = np.array(message) % p
    G = np.array(G) % p
    return (message @ G) % p
