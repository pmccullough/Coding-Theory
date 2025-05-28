import numpy as np  # Import NumPy for matrix operations
from sympy import isprime
class Encoder:
    def __init__(self, G=None, p =2):  
        if not isinstance(p, int) or not isprime(p):
            raise ValueError("p must be a prime number.")
        
        self.p = p

        # If no generator matrix is provided, use default for [3,2] binary code
        if G is None:
                G = np.array([
                    [1,1,0],
                    [0,1,1]
                ])
        else:
            G = np.array(G)  # Convert to NumPy array again for assignment
            if not np.issubdtype(G.dtype, np.integer) or not np.all((0 <= G) & (G < p)):
                raise ValueError(f"All elements of the generator matrix must be integers in [0, {p-1}].")
        
        self.G = G  # Store validated generator matrix
        self.k, self.n = self.G.shape


    def encode(self, message):
        # Validate that the message is a list or array of integers
        if not isinstance(message, (list, np.ndarray)):
            raise ValueError("Message must be a list or numpy array.")
            
        # Convert message to a NumPy array and validate its length
        message = np.array(message)
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k}.")

        # Check that all values are in the valid range
        if not np.issubdtype(message.dtype, np.integer) or not np.all((0 <= message) & (message < self.p)):
            raise ValueError(f"All message values must be integers in [0, {self.p - 1}].")
          
        # Perform encoding using matrix multiplication
        codeword = np.dot(message, self.G) % self.p
        return codeword
# Example usage
if __name__ == "__main__":
    encoder = Encoder()
    message = [1, 0, 1]
    codeword = encoder.encode(message)
    print("Encoded codeword:", codeword)
