import numpy as np 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import poly_add, poly_mul, coeff_mod_p, poly_to_string, poly_mod

class GF:
    def __init__(self, p, m, prim_poly):
        self.p = p
        self.m = m
        self.q = p ** m
        # Allow prim_poly as int or list
        if isinstance(prim_poly, int):
            self.prim_poly = prim_poly
        elif isinstance(prim_poly, (list, tuple)):
            # Convert list of coefficients to integer representation
            self.prim_poly = sum(
                coeff * (p ** i)
                for i, coeff in enumerate(prim_poly)
            )
        else:
            raise ValueError("prim_poly must be int or list/tuple of coefficients")
        self.alpha = [0,1] + [0] * (self.m - 2)  if self.m > 2 else [0, 1]

    def add(self, a, b):
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        sum = poly_add(a,b)
        reduced = coeff_mod_p(sum, self.p)
        return self.to_int(reduced)

    def mul(self, a, b):
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        product = poly_mul(ca, cb)
        reduced = poly_mod(product, self.prim_poly, self.p)
        return self.to_int(reduced)
    
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

    
    def __repr__(self):
        return f"GF({self.p}^{self.m}) with prim_poly={bin(self.prim_poly)}"
    

