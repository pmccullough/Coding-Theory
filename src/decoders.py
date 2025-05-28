import numpy as np
from sympy import isprime   

class Decoder:
    def __init__(self, G, p=2):
        G = np.array(G)  # Ensure input is a NumPy array for matrix operations

        # Validate that all elements are integers in the field F_p = {0, 1, ..., p-1}
        if not np.issubdtype(G.dtype, np.integer) or not np.all((0 <= G) & (G < p)):
            raise ValueError(f"All elements of the generator matrix must be integers in [0, {p-1}].")
        
        self.G = G              # Store the generator matrix
        self.k, self.n = G.shape  # Derive k (dimension) and n (length) from shape
        self.p = p              # Store field size (modulus)

    def decode(self, received):
        raise NotImplementedError("Subclasses must implement decode()")
    
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

# Example usage
if __name__ == "__main__":
    G = [
        [1, 1, 0],
        [0, 1, 1]
    ]
    decoder = Decoder.NearestNeighborDecoder(G)
    
    received = [1, 0, 1]  # Example received vector
    decoded_message = decoder.decode(received)
    
    print("Decoded message:", decoded_message)