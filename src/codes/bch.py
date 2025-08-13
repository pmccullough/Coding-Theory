import numpy as np 
from .gf import GF




class BCHCode:
    def __init__(self, p, m, n, prim_poly, t, b=1):
        """
        p: base field size (2 for binary)
        m: extension degree
        prim_poly: irreducible polynomial for GF(p^m) as list or int
        t: error-correcting capability
        b: starting power for consecutive roots (1 for narrow-sense BCH)
        """
        self.gf = GF(p, m, prim_poly)
        self.n = n
        self.p = p
        self.m = m
        self.t = t
        self.b = b

        # Build generator polynomial
        self.g = self._generate_g()

    def _alpha_pow(self, k):
        """Return α^k as coefficients list using GF multiplication."""
        result = [1] + [0] * (self.m - 1)    # 1
        for _ in range(k % self.n):
            result = self.gf.coeffs(self.gf.mul(result, self.gf.alpha))
        return result

    def _cyclotomic_coset(self, start):
        """Compute cyclotomic coset modulo n starting from 'start'."""
        coset = set()
        x = start % self.n
        while x not in coset:
            coset.add(x)
            x = (x * self.p) % self.n
        return sorted(coset)

    def _minimal_polynomial(self, exp):
        """Minimal polynomial of α^exp over GF(p)."""
        coset = self._cyclotomic_coset(exp)
        poly = [1]  # start with x^0 = 1
        for i in coset:
            alpha_i = self._alpha_pow(i)
            # (x - α^i) in GF
            term = [1, (-1) % self.p] if self.p != 2 else [1, 1]  # p=2 => -α^i = α^i
            term[1] = self.gf.to_int(self.gf.coeffs(alpha_i))
            poly = self._poly_mul(poly, term)
        return poly

    def _poly_mul(self, a, b):
        """Multiply polynomials over GF(p)."""
        res = [0] * (len(a) + len(b) - 1)
        for i in range(len(a)):
            for j in range(len(b)):
                res[i + j] = (res[i + j] + a[i] * b[j]) % self.p
        return res

    def _generate_g(self):
        """Generator polynomial g(x) = LCM of minimal polynomials for α^b,...,α^(b+2t-1)."""
        g = [1]
        seen_cosets = set()
        for i in range(self.b, self.b + 2*self.t):
            coset = tuple(self._cyclotomic_coset(i))
            if coset in seen_cosets:
                continue
            seen_cosets.add(coset)
            mpoly = self._minimal_polynomial(i)
            g = self._poly_mul(g, mpoly)
        return g

    def encode(self, message):
        """Systematic BCH encoding."""
        k = len(message)
        msg_poly = list(message) + [0] * (len(self.g) - 1)
        for i in range(k):
            if msg_poly[i] != 0:
                for j in range(len(self.g)):
                    msg_poly[i + j] = (msg_poly[i + j] - self.g[j]) % self.p
        remainder = msg_poly[k:]
        return list(message) + remainder

if __name__ == "__main__":
    # Example: binary (7, 4, 1) BCH code (Hamming code)
    p = 2
    m = 3
    n = 7
    prim_poly = [1, 1, 0, 1]  # x^3 + x + 1, lowest degree first
    t = 1
    b = 1
    bch = BCHCode(p, m, n, prim_poly, t, b)

    print("Generator polynomial g(x):", bch.gf.poly_str(bch.g))

    # Test encoding all 4-bit messages
    print("\nMessage  |  Codeword")
    for i in range(16):
        msg = [int(x) for x in bin(i)[2:].zfill(4)]
        codeword = bch.encode(msg)
        print(f"{msg}  |  {codeword}")