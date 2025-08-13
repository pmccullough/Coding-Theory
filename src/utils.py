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
    return poly