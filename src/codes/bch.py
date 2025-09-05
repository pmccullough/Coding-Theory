from .gf import GF
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import  cyclotomic_coset, poly_to_string, is_power_of_n, gcd

class BCH:
    def __init__(self, gf, q, delta, b=1, n=None):
        """
        q     : Order of the field our code is over.
        delta : Design distance
        b     : Starting exponent (default=1 for narrow-sense)
        n     : Code length. Defaults to q - 1 (primitive BCH)
        """
        self.q = q
        self.gf = gf
        self.delta = delta
        self.b = b
        self.n = n if n is not None else (self.q - 1)
        self._beta = None
        self._beta_pows = None
        self._validate()

        self.g = self._build_generator()
        self.G = self.generator_matrix()

    def _validate(self):
        if not is_power_of_n(self.gf.q,self.q):
            raise ValueError(f"Your field size {self.gf.q} must be a power of {self.q}")

        if gcd(self.q,self.n) != 1:
            raise ValueError(f"n={self.n} must be coprime to q={self.q}")

        if not (2 <= self.delta <= self.n):
            raise ValueError(f"delta must be between 2 and {self.n}")

        if self.b < 0 or self.b + self.delta - 2 > self.n:
            raise ValueError("Invalid starting exponent for given n, delta")
        
    def _build_generator(self):
        alpha = self.gf.primitive_element
        e = (self.gf.q-1) //self.n
        beta = self.beta

        E = [ (self.b + i) % self.n for i in range(self.delta - 1) ]
        exp_roots = set()
        for i in E:
           exp_roots.update(cyclotomic_coset(self.n, self.q, i))
        
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

        # ---- use GF-provided β ----
    @property
    def beta(self):
        """Primitive n-th root of unity in GF(q), cached."""
        if self._beta is None:
            # raises if n ∤ (q-1); your validate() should enforce this anyway
            self._beta = self.gf.nth_root_of_unity(self.n)
        return self._beta

    def beta_pow(self, k: int):
        """Return β^k with O(1) lookup if the table is built."""
        if self._beta_pows is None:
            # build table: β^0..β^{n-1}
            b = self.beta
            self._beta_pows = [1] * self.n
            for i in range(1, self.n):
                self._beta_pows[i] = self.gf.mul(self._beta_pows[i-1], b)
        return self._beta_pows[k % self.n]
    
    def syndromes(self, y):
        """
        Compute the syndromes S1, S2, ..., S_{delta-1} for the received vector y.
        """
        S = []
        for i in range(1, self.delta):
            x = self.beta_pow(i)
            acc = 0
            for c in reversed(y):
                acc = self.gf.add(self.gf.mul(acc, x), c)
            S.append(acc)
        return S

    def syndrome_poly(self,y):
        """
        Returns the syndrome polynomial S(x) = S_1 + S_2 x + S_3 x^2 + ... + S_{delta-1} x^{delta-2}
        """
        S = self.syndromes(y)
        return [1] + S
    
    def generator_matrix(self):
        """
        Returns the generator matrix G for this BCH code.
        Each row is a cyclic shift of the generator polynomial, padded to length n.
        """
        g = self.g
        n = self.n
        k = n - len(g) + 1
        G = []
        for i in range(k):
            row = [0] * i + g + [0] * (n - len(g) - i)
            G.append([c % self.gf.p for c in row])
        return G
    
    def g_poly_repr(self, var='x'):
        """
        Return a human-readable string for the generator polynomial g(x).
        """
        alpha_coeffs = [self.gf.alpha_power_repr(c) for c in self.g]
        return poly_to_string(alpha_coeffs, var=var)
    
    def s_poly_repr(self,y, var='x'):
        """
        Return a human-readable string for the syndrome polynomial S(x).
        """
        S = self.syndrome_poly(y)
        alpha_coeffs = [self.gf.alpha_power_repr(c) for c in S]
        return poly_to_string(alpha_coeffs, var=var)
    
    @property
    def k(self):
        # deg g = len(g) - 1 for lowest-degree-first coeffs
        return self.n - (len(self.g) - 1)
      
if __name__ == "__main__":
    gf = GF(2,6,[1,1,0,0,0,0,1])
    bch = BCH(gf,q = 2, delta = 3, n=9, b = 1)
    print(f"Generator polynomial: {bch.g_poly_repr()}")