import numpy as np
import sympy as sp  

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
        raise NotImplementedError("Subclasses must implement decode()")
    
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

    def _enumerate_messages(self):
        """
        Generate all possible messages of length k in the field F_p.
        """
        # Generate all combinations of messages in F_p
        return [np.array(msg) for msg in np.ndindex(*(self.p,) * self.k)]
    
    def _encode_message(self, message):
        """
        Internal: Encode a message using the generator matrix G.
        """
        if not isinstance(message, (list, np.ndarray)):
            raise ValueError("Message must be a list or numpy array.")
        
        message = np.array(message)
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k}.")
        
        if not np.issubdtype(message.dtype, np.integer) or not np.all((0 <= message) & (message < self.p)):
            raise ValueError(f"All message values must be integers in [0, {self.p - 1}].")
        
        # Perform encoding using matrix multiplication
        codeword = np.dot(message, self.G) % self.p
        return codeword
    
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
            codeword = self._encode_message(message)
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
            codeword = self._encode_message(message)
            print(f"{message.tolist()}  ->  {codeword.tolist()}")

    def hamming_distance(self, codeword1, codeword2):
        """
        Calculate the Hamming distance between two codewords.
        """
        if len(codeword1) != len(codeword2):
            raise ValueError("Codewords must be of the same length.")
        
        # Count the number of differing positions
        return np.sum(codeword1 != codeword2)
    
class NearestNeighborDecoder(Decoder):
    def __init__(self, G, p=2):
        super().__init__(G, p)
        self.codebook = self._generate_codebook()  

    def _generate_codebook(self):
        """Precompute all (message, codeword) pairs."""
        codebook = []
        for message in self._enumerate_messages():
            codeword = self._encode_message(message)
            codebook.append((message, codeword))
        return codebook
        
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
        
        # Find closest codeword by Hamming distance
        min_distance = self.n + 1
        closest_message = None

        for message, codeword in self.codebook:
            dist = self.hamming_distance(received, codeword)
            if dist < min_distance:
                min_distance = dist
                closest_message = message

        return closest_message


class SyndromeDecoder(Decoder):
    def __init__(self, G, p=2):
        super().__init__(G, p)
        if self.H is None:
            raise ValueError("Parity-check matrix H required for syndrome decoding.")
        self.syndrome_table = self._build_syndrome_table()


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






# Example usage
if __name__ == "__main__":
     # Generator matrix in standard form: [I_k | P]
    G = np.array([
        [1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1]
    ])

    decoder = Decoder(G, p=2)

    decoder.describe()