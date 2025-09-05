import numpy as np 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import poly_add, poly_mul, coeff_mod_p, poly_to_string, poly_mod, superscript

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

        self._build_log_tables()

    def add(self, a, b):
        if self.m == 1:
            return (a + b) % self.p
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        sum = poly_add(ca, cb)
        reduced = coeff_mod_p(sum, self.p)
        return self.to_int(reduced)
    
    def mul(self, a, b):
        """
        Fast multiplication using log/antilog tables.
        Only valid for nonzero a, b.
        """
        a = self.to_int(a)
        b = self.to_int(b)
        if a == 0 or b == 0:
            return 0
        if not hasattr(self, "log") or not hasattr(self, "antilog"):
            raise AttributeError("Log/antilog tables not built. Call build_log_tables() first.")
        k = (self.log[a] + self.log[b]) % (self.q - 1)
        return self.antilog[k]
    
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
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            return coeffs
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

    def scalar_mul_base(self, a, k_mod_p: int):
        """
        Multiply field element 'a' by the base-field scalar k (0..p-1).
        This is NOT general field multiplication by an arbitrary GF(q) element;
        it's k times 'a' where k is an integer modulo p.
        """
        k = k_mod_p % self.p
        if not isinstance(a, int):
            a = self.to_int(a)
        if k == 0 or a == 0:
            return 0
        if k == 1:
            return a
        # Simple and robust: repeated addition (p is tiny in practice)
        res = 0
        for _ in range(k):
            res = self.add(res, a)
        return res

    def nth_root_of_unity(self,n):
        """ Returns a primitive n-th root of unity in GF(p^m). """
        if (self.q - 1) % n != 0:
            raise ValueError(f"n={n} does not divide q^m - 1")
        e = (self.q - 1) // n
        return self.pow(self.primitive_element, e)

    def poly_repr(self, a, var='α'):
        """
        Return a human-readable string for field element a as a polynomial.
        Accepts either int or list as input.
        """
        coeffs = self.coeffs(a)
        return poly_to_string(coeffs, var=var)
    
    def int_repr(self, a):
        """
        Return integer representation of field element a.
        Accepts either int or list as input.
        """
        return self.to_int(self.coeffs(a))
    
    def alpha_power_repr(self, a):
        """
        Return a string representing the field element 'a' as a power of the primitive element α,
        e.g., 'α¹²', or '0' for zero. Accepts int or list.
        """
        a_int = self.to_int(a)
        if a_int == 0:
            return "0"
        k = self.log.get(a_int, None)
        if k == 0:
            return "1"
        if k == 1:
            return "α"
        if k is not None:
            return f"α{superscript(k)}"
        # Fallback: show as polynomial
        return self.poly_repr(a_int, var='α')
    
    def print_table(self):
        """
        Print all elements of GF(p^m) in order of increasing exponent in α representation:
        - Integer representation (via int_repr)
        - Binary/ternary/base-p representation
        - Primitive element power (via alpha_power_repr)
        - Polynomial representation (via poly_repr)
        """
        if self.p == 2:
            base = "Binary"
        elif self.p == 3:
            base = "Ternary"
        else:
            base = f"Base-{self.p}"
        print(f"{'Int':>3} | {base:>8} | {'α^k':>8} | {'Polynomial':>20}")
        print("-" * 50)
        # Print zero element first
        zero_int = self.int_repr(0)
        zero_bin = bin(zero_int)[2:].zfill(self.m)
        zero_alpha = self.alpha_power_repr(0)
        zero_poly = self.poly_repr(0, var='α')
        print(f"{zero_int:>3} | {zero_bin:>8} | {zero_alpha:>8} | {zero_poly:>20}")
        # Then print nonzero elements in α^k order
        for k in range(self.q - 1):
            i = self.antilog[k]
            int_str = self.int_repr(i)
            bin_str = bin(int_str)[2:].zfill(self.m)
            alpha_str = self.alpha_power_repr(i)
            poly_str = self.poly_repr(i, var='α')
            print(f"{int_str:>3} | {bin_str:>8} | {alpha_str:>8} | {poly_str:>20}")

    def _mul_poly(self, a, b):
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
    
    def _build_log_tables(self):
        """
        Builds log and antilog tables for the multiplicative group of GF(q).
        - antilog[k] = α^k (as int), for k in 0..q-2
        - log[a] = k such that a = α^k, for a in 1..q-1
        """
        antilog = [1]
        for k in range(1, self.q-1):
            antilog.append(self._mul_poly(antilog[-1], self.primitive_element))
        log = {a: k for k, a in enumerate(antilog)}
        self.antilog = antilog
        self.log = log

    def __repr__(self):
        return f"GF({self.p}^{self.m}) with primitive_polynomial={self.primitive_polynomial}"


if __name__ == "__main__":
    gf16 = GF(2, 4, [1, 0, 0, 1, 1])
    gf16.print_table()