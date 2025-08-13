import numpy as np 

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
        result_coeffs = [(x + y) % self.p for x, y in zip(ca, cb)]
        return self.to_int(result_coeffs)
       

    def mul(self, a, b):
        ca = self.coeffs(a)
        cb = self.coeffs(b)
        m = self.m
        p = self.p
        # Polynomial multiplication (convolution)
        prod = [0] * (2 * m - 1)
        for i in range(m):
            for j in range(m):
                prod[i + j] = (prod[i + j] + ca[i] * cb[j]) % p
        # Primitive polynomial coefficients (lowest degree first, length m+1)
        prim_coeffs = self.coeffs(self.prim_poly)
        if len(prim_coeffs) < m + 1:
            prim_coeffs += [0] * (m + 1 - len(prim_coeffs))
        # Modular reduction
        for i in range(len(prod) - 1, m - 1, -1):
            if prod[i]:
                for j in range(m + 1):
                    prod[i - m + j] = (prod[i - m + j] - prod[i] * prim_coeffs[j]) % p
                prod[i] = 0
        result_coeffs = prod[:m]
        return self.to_int(result_coeffs)
    
    def build_log_tables(self):
        """
        Build log and antilog tables for the field.
        Returns:
            log_table: dict mapping tuple(coeffs) -> exponent (0 to q-2)
            antilog_table: list mapping exponent -> tuple(coeffs)
        """
        current = [1] + [0] * (self.m - 1)   # 1
        log_table = {}
        antilog_table = []
        for k in range(self.q - 1):
            tup = tuple(current)
            log_table[tup] = k
            antilog_table.append(tup)
            current = self.coeffs(self.mul(current, self.alpha))
        return log_table, antilog_table
    
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
        
    def print_elements_table(self):
        """
        Print a table of all nonzero elements in the field, ordered by increasing alpha power.
        Columns: Binary | Alpha Power | Polynomial (list)
        """
        current = [1]+ [0] * (self.m - 1) 

        seen = set()
        elements = []
        for k in range(self.q - 1):
            poly = self.coeffs(current)
            bin_str = bin(self.to_int(poly))[2:].zfill(self.m)
            elements.append((bin_str, f"α^{k}", poly))
            seen.add(tuple(poly))
            current = self.coeffs(self.mul(current, self.alpha))
        print(f"{'Binary':>8} | {'Alpha Power':>10} | {'Polynomial':>15}")
        print("-" * 40)
        for bin_str, alpha_str, poly in elements:
            print(f"{bin_str:>8} | {alpha_str:>10} | {self.poly_str(poly):>15}")

    def poly_str(self, coeffs):
        """
        Return a string representing the element as a polynomial in x.
        Example: [1, 0, 1] -> '1 + x^2'
        """
        coeffs = self.coeffs(coeffs)
        terms = []
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{'' if c == 1 else c}\u03B1")
            else:
                terms.append(f"{'' if c == 1 else c}\u03B1^{i}")
        if not terms:
            return "0"
        return " + ".join(terms)
    
    def __repr__(self):
        return f"GF({self.p}^{self.m}) with irr_poly={bin(self.irr_poly)}"
    
if __name__ == "__main__":
    gf8 = GF(2, 3, [1, 1, 0, 1])
    a = [0, 1, 0]  # α
    b = [0, 1, 1]  # α + α^2
    print(gf8.poly_str(a), "*(", gf8.poly_str(b), ")=", gf8.poly_str(gf8.mul(a, b)))
