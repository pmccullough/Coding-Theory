import numpy as np
from .base import Decoder
from src.utils import hamming_weight, hamming_distance

class SyndromeDecoder(Decoder):
    def __init__(self, G, p=2):
        super().__init__(G, p)
        if self.H is None:
            raise ValueError("Parity-check matrix H required for syndrome decoding.")
        self.syndrome_table = self._build_syndrome_table()
    
    def decode(self, received):
        """
        Decode the received vector using syndrome decoding.
        Returns the most likely message.
        """
        received = np.array(received)
        if len(received) != self.n:
            raise ValueError(f"Received vector must be length {self.n}")
        if not np.issubdtype(received.dtype, np.integer) or not np.all((0 <= received) & (received < self.p)):
            raise ValueError(f"All received values must be integers in [0, {self.p - 1}].")  
        
        # Calculate the syndrome
        syndrome = (self.H @ received.T) % self.p
        error = self.syndrome_table.get(tuple(syndrome), np.zeros(self.n, dtype=int))
        corrected = (received - error) % self.p
        return corrected 

    def _build_syndrome_table(self):
        """Map each syndrome to a likely error vector (weight-1 only for now)."""
        table = {}
        n = self.n

        for i in range(n):
            for val in range(1, self.p):
                e = np.zeros(n, dtype=int)
                e[i] = val
                syndrome = (self.H @ e.T) % self.p
                syndrome_key = tuple(syndrome)
                if syndrome_key not in table:
                    table[syndrome_key] = e

        return table
    
if __name__ == "__main__":
    # Example [7,4] code over F_2 in systematic form
    G = [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ]
    decoder = SyndromeDecoder(G, p=2)
    message = np.array([1, 0, 1, 1])
    from src.utils import encode
    codeword = encode(message, G, 2)
    print("Original message:", message)
    print("Encoded codeword:", codeword)

    # Introduce a single-bit error
    received = codeword.copy()
    received[4] ^= 1  # flip bit 4
    print("Received (with error):", received)

    # Decode
    corrected = decoder.decode(received)
    print("Corrected codeword:", corrected)