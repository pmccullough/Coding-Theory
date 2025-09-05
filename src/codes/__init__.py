from .gf import GF
from .bch import BCH

GF4  = GF(2, 2, [1, 1, 1])           # x^2 + x + 1
GF8  = GF(2, 3, [1, 1, 0, 1])        # x^3 + x + 1
GF16 = GF(2, 4, [1, 0, 0, 1, 1])     # x^4 + x^3 + 1
GF32 = GF(2, 5, [1, 0, 1, 0, 0, 1])  # x^5 + x^2 + 1
GF64 = GF(2, 6, [1, 1, 0, 0, 0, 0, 1]) # x^6 + x + 1