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

def poly_add(a, b):
    """
    Add two polynomials over GF(p).
    a, b: lists of coefficients (lowest degree first).
    Returns sum in lowest degree first order.
    """
    max_len = max(len(a), len(b))
    result = [0] * max_len
    for i in range(max_len):
        coeff_a = a[i] if i < len(a) else 0
        coeff_b = b[i] if i < len(b) else 0
        result[i] = (coeff_a + coeff_b)
    return result

def poly_mul(a, b):
    """
    Multiply two polynomials a and b over GF(p).
    a, b: lists of coefficients (lowest degree first).
    Returns product in lowest degree first order.
    """
    deg_a = len(a)
    deg_b = len(b)
    prod = [0] * (deg_a + deg_b - 1)
    for i in range(deg_a):
        for j in range(deg_b):
            prod[i + j] = (prod[i + j] + a[i] * b[j])
    return prod

def coeff_mod_p(poly, p):
    """Return a copy with all coeffs reduced mod p (lowest degree first)."""
    return [c % p for c in poly]

def poly_to_string(poly, var='x'):
    """
    Convert a polynomial (lowest degree first) to a human-readable string.
    poly: list of coefficients (lowest degree first)
    var: variable name for display, e.g., 'x' or 'α'
    """
    terms = []
    for power, coeff in enumerate(poly):
        if coeff == 0:
            continue
        if power == 0:
            term = f"{coeff}"
        elif power == 1:
            term = f"{coeff}{var}" if coeff != 1 else f"{var}"
        else:
            term = f"{coeff}{var}^{power}" if coeff != 1 else f"{var}^{power}"
        terms.append(term)
    return " + ".join(reversed(terms)) if terms else "0"

def poly_mod(poly, mod_poly, p):
    """
    Reduce poly modulo mod_poly over GF(p).
    Both poly and mod_poly are lists of coefficients (lowest degree first).
    Returns the remainder (lowest degree first, degree < len(mod_poly)-1).
    """
    poly = poly[:]  # Make a copy
    mod_deg = len(mod_poly) - 1
    while len(poly) >= len(mod_poly):
        lead_coeff = poly[-1] % p
        if lead_coeff != 0:
            for i in range(len(mod_poly)):
                poly[-len(mod_poly) + i] = (poly[-len(mod_poly) + i] - lead_coeff * mod_poly[i]) % p
        poly.pop()  # Remove highest degree term
    return coeff_mod_p(poly, p)

def sieve(n):
   
    #Create a boolean list to track prime status of numbers
    prime = [True] * (n + 1)
    p = 2

    # Sieve of Eratosthenes algorithm
    while p * p <= n:
        if prime[p]:
            
            # Mark all multiples of p as non-prime
            for i in range(p * p, n + 1, p):
                prime[i] = False
        p += 1

    # Collect all prime numbers
    res = []
    for p in range(2, n + 1):
        if prime[p]:
            res.append(p)
    
    return res

def isprimepower(n):
    """
    Check if a number is a prime power (p^k for some prime p and integer k >= 1).
    """
    if n < 2:
        return False
    for p in sieve(int(n**0.5) + 1):
        power = p
        while power < n:
            power *= p
        if power == n:
            return True
    return False

def gcd(a, b):
    """
    Compute the greatest common divisor of a and b using the Euclidean algorithm.
    """
    while b != 0:
        a, b = b, a % b
    return a

def cyclotomic_coset(n,q,s):
    """
    Compute the s-th cyclotomic coset of n mod q.
    """
    if not isprimepower(q):
        raise ValueError("q must be a prime power.")
    if gcd(n, q) != 1:
        raise ValueError("n must be coprime to q.")
    if s < 0 or s > n - 1:
        raise ValueError("s must be in the range [0, n-1).")
    coset = []
    k = s
    while k not in coset:
        coset.append(k)
        k = (k * q) % n
    return coset

def cyclotomic_cosets(n, q):
    """
    Compute all cyclotomic cosets of n mod q.
    """
    if not isprimepower(q):
        raise ValueError("q must be a prime power.")
    if gcd(n, q) != 1:
        raise ValueError("n must be coprime to q.")
    cosets = []
    for s in range(n):
        cosets.append(cyclotomic_coset(n, q, s))
    return cosets

if __name__ == "__main__":
    # Example usage
    cosets = cyclotomic_cosets(9, 2)
    for i, coset in enumerate(cosets):
        print(f"Coset {i}: {coset}")