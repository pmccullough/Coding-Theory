from .gf import GF
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import  cyclotomic_coset, poly_to_string

class BCH:
    def __init__(self, gf, delta, b=1, n=None):
        """
        gf    : GF object for GF(p^m), with gf.q = p^m
        delta : Design distance
        b     : Starting exponent (default=1 for narrow-sense)
        n     : Code length. Defaults to q - 1 (primitive BCH)
        """
        self.gf = gf
        self.delta = delta
        self.b = b
        self.n = n if n is not None else (gf.q - 1)

        self._validate()
        self.g = self._build_generator()

    def _validate(self):
        # Ensure n divides p^m - 1, so there is an element of order n in GF(q)
        if (self.gf.q - 1) % self.n != 0:
            raise ValueError(f"n={self.n} does not divide q^m - 1")

        if not (2 <= self.delta <= self.n):
            raise ValueError(f"delta must be between 2 and {self.n}")

        if self.b < 0 or self.b + self.delta - 2 >= self.n:
            raise ValueError("Invalid starting exponent for given n, delta")
        
    def _build_generator(self):
        alpha = self.gf.primitive_element
        e = (self.gf.q-1) //self.n
        beta = self.gf.pow(alpha,e)

        E = [ (self.b + i) % self.n for i in range(self.delta - 1) ]
        exp_roots = set()
        for i in E:
           exp_roots.update(cyclotomic_coset(self.n, self.gf.p, i))
        
        generator = [1]
        for e in exp_roots:
            root = self.gf.pow(beta, e)
            term = [self.gf.neg(root),1]
            generator = self._poly_mul(generator, term)
        return self._poly_monic(generator)

    def _poly_mul(self, a, b):
        gf = self.gf
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0: 
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                out[i + j] = gf.add(out[i + j], gf.mul(ai, bj))
        return self._poly_trim(out)

    def _poly_trim(self, p):
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        return p

    def _poly_monic(self, p):
        p = self._poly_trim(p[:])
        lc = p[-1]
        if lc == 1 or lc == 0:
            return p
        inv_lc = self._inv(lc)
        return [ self.gf.mul(c, inv_lc) for c in p ]   

if __name__ == "__main__":
    gf = GF(2,6,[1,1,0,0,0,0,1])
    bch = BCH(gf, 3, n=9, b = 1)
    print(f"Generator polynomial: {poly_to_string(bch.g)}")