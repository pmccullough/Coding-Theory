import numpy as np  # Import NumPy for matrix operations
from sympy import isprime
class Encoder:
    def __init__(self, n=6, k=3, p=2, G=None):  
        # Validate that n (codeword length) is a positive integer
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        
        # Validate that k (dimension of the code) is a positive integer
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        
        # Validate that p (field size) is an integer greater than 1
        if not isinstance(p, int) or not isprime(p):
            raise ValueError("p must be a prime number.")
        
        # If a generator matrix is provided, validate shape and values
        if G is not None:
            G = np.array(G)
            if G.shape != (k, n):
                raise ValueError("G must be of shape (k, n).")
            if not np.issubdtype(G.dtype, np.integer) or not np.all((0 <= G) & (G < p)):
                raise ValueError(f"All elements of G must be integers in [0, {p-1}].")
        
        # Store parameters
        self.n = n                # Length of the codeword
        self.k = k                # Dimension of the code (number of message symbols)
        self.p = p                # Field size

        # If no generator matrix is provided, use default for [6,3] binary code
        if G is None:
            if n == 6 and k == 3 and p == 2:
                G = np.array([
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1]
                ])
            else:
                raise ValueError("No default matrix available for these parameters.")
        else:
            G = np.array(G)  # Convert to NumPy array again for assignment
            if G.shape != (k, n):
                raise ValueError(f"Provided generator matrix must have shape ({k}, {n})")
            if not np.issubdtype(G.dtype, np.integer) or not np.all((0 <= G) & (G < p)):
                raise ValueError(f"All elements of the generator matrix must be integers in [0, {p-1}].")
        
        self.G = G  # Store validated generator matrix
        
    def encode(self, message):
        # Validate that the message is a list or array of integers
        if not isinstance(message, (list, np.ndarray)):
            raise ValueError("Message must be a list or numpy array.")
            
        # Convert message to a NumPy array and validate its length
        message = np.array(message) % self.p
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k}.")
            
        # Perform encoding using matrix multiplication
        codeword = np.dot(message, self.G) % self.p
        return codeword
# Example usage
if __name__ == "__main__":
    encoder = Encoder()
    message = [1, 0, 1]
    codeword = encoder.encode(message)
    print("Encoded codeword:", codeword)
