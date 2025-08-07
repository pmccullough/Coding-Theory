import numpy as np 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import encode, hamming_distance

class Decoder:
    def __init__(self, G, p=2):
        G = np.array(G)  # Ensure input is a NumPy array for matrix operations

        # Validate that all elements are integers in the field F_p = {0, 1, ..., p-1}
        if not np.issubdtype(G.dtype, np.integer) or not np.all((0 <= G) & (G < p)):
            raise ValueError(f"All elements of the generator matrix must be integers in [0, {p-1}].")
        
        self.G = G              # Store the generator matrix
        self.k, self.n = G.shape  # Derive k (dimension) and n (length) from shape
        self.p = p              # Store field size (modulus)
        self.H = self._compute_parity_check_matrix()  # Compute parity-check matrix H

    def decode(self, received):
        """
        Return the message whose codeword is closest to the received vector.
        """
        received = np.array(received)  # Ensure NumPy array
        
        # Input validation
        if len(received) != self.n:
            raise ValueError(f"Received vector must be length {self.n}")
        if not np.issubdtype(received.dtype, np.integer) or not np.all((0 <= received) & (received < self.p)):
            raise ValueError(f"All received values must be integers in [0, {self.p - 1}].")
        
        codebook = self._generate_codebook()  # Precompute codebook if not already done
        # Find closest codeword by Hamming distance
        min_distance = self.n + 1
        closest_message = None

        for message, codeword in codebook:
            dist = hamming_distance(received, codeword)
            if dist < min_distance:
                min_distance = dist
                closest_message = message

        return closest_message
    
    def describe(self):
        print(f"Linear [{self.n}, {self.k}] code over 𝔽_{self.p}")
        
        self.display_G()
        self.display_H()

        print("\nFirst few codewords:")
        max_preview = min(8, self.p**self.k)
        for i, message in enumerate(self._enumerate_messages()):
            if i >= max_preview:
                print("...")
                break
            codeword = encode(message,self.G, self.p)
            print(f"  {message} → {codeword}")

    def display_G(self):
        """Display the generator matrix G."""
        print("Generator Matrix G:")
        print(self.G)

    def display_H(self):
        """Display the parity-check matrix H."""
        if self.H is not None:
            print("Parity-Check Matrix H:")
            print(self.H)
        else:
            print("Parity-Check Matrix H is not defined (G is not in systematic form).")

    def display_n(self):
        """Display the codeword length n."""
        print(f"n (codeword length): {self.n}")

    def display_k(self):
        """Display the message length k."""
        print(f"k (message length): {self.k}")

    def display_p(self):
        """Display the field size p."""
        print(f"p (field size): {self.p}")

    def display_code(self):
        """
        Print all possible messages and their corresponding codewords for this code.
        """
        print("Message  ->  Codeword")
        for message in self._enumerate_messages():
            codeword = encode(message, self.G,self.p)
            print(f"{message.tolist()}  ->  {codeword.tolist()}")

    def _enumerate_messages(self):
        """
        Generate all possible messages of length k in the field F_p.
        """
        # Generate all combinations of messages in F_p
        return [np.array(msg) for msg in np.ndindex(*(self.p,) * self.k)]
    
    def _generate_codebook(self):
        """Precompute all (message, codeword) pairs."""
        codebook = []
        for message in self._enumerate_messages():
            codeword = encode(message, self.G, self.p)
            codebook.append((message, codeword))
        return codebook
    
    def _compute_parity_check_matrix(self):
        """
        Construct the parity-check matrix H assuming G = [I_k | P]
        Returns H or None if G is not in systematic form.
        """
        I_k = np.eye(self.k, dtype=int)
        if not np.array_equal(self.G[:, :self.k], I_k):
            return None  # Not in systematic form

        P = self.G[:, self.k:]
        PT = P.T
        I_nk = np.eye(self.n - self.k, dtype=int)
        H = np.concatenate((PT, I_nk), axis=1) % self.p
        return H
    

# Example usage
if __name__ == "__main__":
    G = np.array([
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1]
    ])

    decoder = Decoder(G, p=2)
    decoder.describe()

    # Sample codeword
    msg = [1, 0, 1]
    codeword = encode(msg, G, 2)
    print(f"\nMessage: {msg} → Codeword: {codeword}")

    # Add a 1-bit error
    received = codeword.copy()
    received[2] ^= 1
    print(f"Received: {received}")

    # Decode using brute-force nearest neighbor
    decoded = decoder.decode(received)
    print(f"Decoded message: {decoded}")
   