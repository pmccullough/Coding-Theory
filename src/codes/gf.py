import numpy as np 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import poly_add, poly_mul, coeff_mod_p, poly_to_string, poly_mod

class GF:
    def __init__(self, p, m, primitive_polynomial):
        self.p = p
        self.m = m
        self.q = p ** m
        # Allow primitive_polynomial as int or list
        if isinstance(primitive_polynomial, (list, tuple)):
            self.primitive_polynomial = primitive_polynomial
        else:
            raise ValueError("primitive_polynomial must list/tuple of coefficients")
        self.primitive_element = [0] * self.m
        if self.m > 1:
            self.primitive_element[1] = 1

    def add(self, a, b):
        if self.m == 1:
            return (a + b) % self.p
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        sum = poly_add(ca, cb)
        reduced = coeff_mod_p(sum, self.p)
        return self.to_int(reduced)

    def mul(self, a, b):
        if self.m == 1:
            return (a * b) % self.p
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        product = poly_mul(ca, cb)
        reduced = poly_mod(product, self.primitive_polynomial, self.p)
        # Ensure reduced is length m
        if len(reduced) < self.m:
            reduced += [0] * (self.m - len(reduced))
        elif len(reduced) > self.m:
            reduced = reduced[:self.m]
        return self.to_int(reduced)
    
    def pow(self,a,k):
        result = 1
        for _ in range(k):
            result = self.mul(result, a)
        return result

    def coeffs(self, a):
        # Returns [a_0, ..., a_{m-1}]
        if isinstance(a, int):
            coeffs = []
            x = a
            for i in range(self.m):
                coeffs.append(x % self.p)
                x //= self.p
            return coeffs
        elif isinstance(a, (list, tuple, np.ndarray)):
            a = list(a)
            if len(a) < self.m:
                return a + [0] * (self.m - len(a))
            elif len(a) > self.m:
                return a[:self.m]
            else:
                return a
        else:
            raise ValueError("Input must be int or list-like.")

    def to_int(self, coeffs):
        # [a_0, ..., a_{m-1}]
        coeffs = list(coeffs)
        if len(coeffs) < self.m:
            coeffs = coeffs + [0] * (self.m - len(coeffs))
        return sum(c * (self.p ** i) for i, c in enumerate(coeffs))

    def neg(self, a):
        """
        Additive inverse in GF(p^m).
        - char 2:   -a == a
        - general p: fallback to (p-1) * a using repeated addition,
                     or use sub(0, a) if you already have sub().
        """
        if a == 0:
            return 0
        if self.p == 2:
            return a  # in characteristic 2, a == -a

        # Fast path if you already have subtraction:
        if hasattr(self, "sub"):
            return self.sub(0, a)

        # Generic, O(p) fallback (p is small in practice):
        res = 0
        for _ in range(self.p - 1):  # (p-1) * a
            res = self.add(res, a)
        return res

    def inv(self, a):
        """
        Multiplicative inverse in GF(p^m).
        - Uses Fermat's little theorem in GF(q): a^(q-1) = 1 for a != 0,
          so a^(-1) = a^(q-2).
        - Optional fast path via log/exp tables if present.
        """
        if a == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse in a field.")

        # If you maintain discrete log / antilog tables:
        if hasattr(self, "log") and hasattr(self, "exp"):
            # log[0] is typically undefined; we've excluded a==0 above.
            return self.exp[(-self.log[a]) % (self.q - 1)]

        # Generic path via exponentiation in the multiplicative group of size (q-1)
        return self.pow(a, self.q - 2)

    def sub(self, x, y):
        """ Subtraction as x + (-y). """
        return self.add(x, self.neg(y))
    
    def __repr__(self):
        return f"GF({self.p}^{self.m}) with primitive_polynomial={self.primitive_polynomial}"
