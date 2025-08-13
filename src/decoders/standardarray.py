import numpy as np
from .base import Decoder
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import encode, hamming_weight

class StandardArrayDecoder(Decoder):
    """
    Implements standard-array (complete) decoding for linear codes via coset partitioning.
    """
    def __init__(self, G, p=2):
        super().__init__(G, p)
        # Precompute codebook: list of codewords
        self.codewords = [cw for (_, cw) in self._enumerate_codebook()]
        # Compute coset leaders
        self.coset_leaders = self._compute_coset_leaders()
        # Build standard array as list of cosets (each a list of codewords)
        self.standard_array = self._build_standard_array()

    def decode(self, received):
        """Find the unique coset containing `received` and return its codeword."""
        arr = np.asarray(received)
        if arr.shape != (self.n,):
            raise ValueError(f"Received must be length {self.n}, got {arr.shape}")
        # locate coset by subtracting each leader
        for leader in self.coset_leaders:
            candidate = (arr - leader) % self.p
            if any(np.array_equal(candidate, cw) for cw in self.codewords):
                return candidate
        raise ValueError("Unable to decode: no matching coset found.")
    
    def describe(self):
        super().describe()
        print()
        self.display_coset_leaders()

    def display_coset_leaders(self):
        """Print all coset leaders, one per line."""
        print("Coset leaders:")
        for i, leader in enumerate(self.coset_leaders):
            print(f"{i}: {leader.tolist()}")
    
    def display_standard_array(self):
        """Display the standard array."""
        print("Standard Array:")
        for i, row in enumerate(self.standard_array):
            print(f"Leader {i}:")
            for cw in row:
                print(f"  {cw.tolist()}")

    def _enumerate_codebook(self):
        """Yield all (message, codeword) pairs."""
        for msg in self._enumerate_messages():
            yield msg, encode(msg, self.G, self.p)

    def _compute_coset_leaders(self):
        """
        Find one coset leader per coset by ascending weight then lex-order.
        """
        n = self.n
        p = self.p
        # generate all vectors in F^n sorted by Hamming weight then lex
        all_vecs = [np.array(v) for v in np.ndindex(*([p]*n))]
        # sort by weight, then by tuple
        all_vecs.sort(key=lambda v: (hamming_weight(v), tuple(v.tolist())))
        leaders = []
        seen = set()
        codeword_set = {tuple(cw.tolist()) for cw in self.codewords}
        for v in all_vecs:
            vt = tuple(v.tolist())
            if vt in seen:
                continue
            # this v starts a new coset
            leaders.append(v)
            # mark all members of coset v+C
            for cw in self.codewords:
                coset_vec = tuple(((v + cw) % p).tolist())
                seen.add(coset_vec)
            if len(leaders) == p**(self.n - self.k):
                break
        return leaders

    def _build_standard_array(self):
        """Return a list of cosets; each is a list of codewords shifted by its leader."""
        p = self.p
        cosets = []
        for leader in self.coset_leaders:
            row = []
            for cw in self.codewords:
                row.append((leader + cw) % p)
            cosets.append(row)
        return cosets

if __name__ == "__main__":
    # Example [7,4] code over F_2 in systematic form
    G = [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ]
    decoder = StandardArrayDecoder(G, p=2)
    from src.utils import encode
    import numpy as np

    message = np.array([1, 0, 1, 1])
    codeword = encode(message, G, 2)
    print("Original message:", message)
    print("Encoded codeword:", codeword)

    # Introduce a single-bit error
    received = codeword.copy()
    received[4] ^= 1  # flip bit 4
    print("Received (with error):", received)

    # Decode
    decoded = decoder.decode(received)
    print("Decoded codeword:", decoded)  