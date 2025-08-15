# tests/test_gf.py
import random
import itertools
import pytest

from codes.gf import GF  


# ---------- Helpers (pure-Python reference ops) ----------

def int_to_coeffs(a: int, p: int, m: int):
    """Base-p expansion of a into m coefficients [a0, ..., a_{m-1}]."""
    coeffs = []
    x = a
    for _ in range(m):
        coeffs.append(x % p)
        x //= p
    return coeffs

def coeffs_to_int(coeffs, p: int):
    """Inverse of int_to_coeffs (assumes len(coeffs) is m)."""
    return sum(c * (p ** i) for i, c in enumerate(coeffs))

def poly_add_mod_p(a_coeffs, b_coeffs, p: int):
    """Coefficient-wise add (lowest degree first), reduced mod p."""
    L = max(len(a_coeffs), len(b_coeffs))
    out = []
    for i in range(L):
        ai = a_coeffs[i] if i < len(a_coeffs) else 0
        bi = b_coeffs[i] if i < len(b_coeffs) else 0
        out.append((ai + bi) % p)
    # trim trailing zeros is not strictly required for addition checks
    return out

def poly_mul_mod_p(a_coeffs, b_coeffs, p: int):
    """Convolution multiply in GF(p)[x]."""
    out = [0] * (len(a_coeffs) + len(b_coeffs) - 1)
    for i, ai in enumerate(a_coeffs):
        for j, bj in enumerate(b_coeffs):
            out[i + j] = (out[i + j] + ai * bj) % p
    return out

def poly_mod(poly, mod_poly, p: int):
    """
    Reduce polynomial (list, lowest degree first) modulo mod_poly over GF(p).
    mod_poly is assumed monic with deg >= 1.
    """
    poly = poly[:]  # copy
    while len(poly) >= len(mod_poly):
        lead = poly[-1] % p
        if lead != 0:
            # subtract lead * x^(k) * mod_poly
            k = len(poly) - len(mod_poly)
            for i in range(len(mod_poly)):
                poly[k + i] = (poly[k + i] - lead * mod_poly[i]) % p
        poly.pop()  # drop highest term
    # ensure length < len(mod_poly)
    return poly

def expected_add(a: int, b: int, p: int, m: int):
    """
    In GF(p^m) with the usual vector-space representation,
    addition is component-wise mod p. No primitive polynomial reduction needed.
    """
    ac = int_to_coeffs(a, p, m)
    bc = int_to_coeffs(b, p, m)
    sc = [(x + y) % p for x, y in zip(ac, bc)]
    return coeffs_to_int(sc, p)

def expected_mul(a: int, b: int, p: int, m: int, prim_poly):
    """
    Multiply as polynomials over GF(p), then reduce mod prim_poly, then
    convert back to the integer packing.
    """
    ac = int_to_coeffs(a, p, m)
    bc = int_to_coeffs(b, p, m)
    prod = poly_mul_mod_p(ac, bc, p)
    red = poly_mod(prod, prim_poly, p)
    # pad to length m
    if len(red) < m:
        red = red + [0] * (m - len(red))
    else:
        red = red[:m]
    return coeffs_to_int(red, p)


# ---------- Fields to test ----------
# NOTE: primitive polynomials are given with lowest degree first.
FIELDS = [
    # Prime fields (m = 1). Using mod_poly = [0, 1] effectively reduces to degree < 1.
    pytest.param(2, 1, [0, 1], id="GF(2)"),
    pytest.param(3, 1, [0, 1], id="GF(3)"),
    pytest.param(5, 1, [0, 1], id="GF(5)"),

    # Binary extensions
    pytest.param(2, 2, [1, 1, 1], id="GF(2^2)  x^2 + x + 1"),            # primitive
    pytest.param(2, 3, [1, 1, 0, 1], id="GF(2^3)  x^3 + x + 1"),          # primitive
    pytest.param(2, 4, [1, 1, 0, 0, 1], id="GF(2^4)  x^4 + x + 1"),       # primitive

    # Ternary extension
    pytest.param(3, 2, [1, 0, 1], id="GF(3^2)  x^2 + 1 (irreducible)"),
]


# ---------- Tests ----------

@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_basic_properties(p, m, prim):
    gf = GF(p, m, prim)
    q = p ** m

    # Zero and one
    zero = 0
    one = 1

    # Additive identity
    for a in range(q):
        assert gf.add(a, zero) == a, f"add({a},0) failed in GF({p}^{m})"
        assert gf.add(zero, a) == a, f"add(0,{a}) failed in GF({p}^{m})"

    # Multiplicative identity
    for a in range(q):
        assert gf.mul(a, one) == a, f"mul({a},1) failed in GF({p}^{m})"
        assert gf.mul(one, a) == a, f"mul(1,{a}) failed in GF({p}^{m})"

    # Closure (range check)
    for a, b in itertools.product(range(q), repeat=2):
        s = gf.add(a, b)
        t = gf.mul(a, b)
        assert 0 <= s < q, f"add result out of range: {s} in GF({p}^{m})"
        assert 0 <= t < q, f"mul result out of range: {t} in GF({p}^{m})"


@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_addition_matches_reference(p, m, prim):
    gf = GF(p, m, prim)
    q = p ** m
    for a, b in itertools.product(range(q), repeat=2):
        expect = expected_add(a, b, p, m)
        got = gf.add(a, b)
        assert got == expect, f"GF({p}^{m}) add mismatch: {a}+{b} -> {got} != {expect}"


@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_multiplication_matches_reference(p, m, prim):
    gf = GF(p, m, prim)
    q = p ** m
    for a, b in itertools.product(range(q), repeat=2):
        expect = expected_mul(a, b, p, m, prim)
        got = gf.mul(a, b)
        assert got == expect, f"GF({p}^{m}) mul mismatch: {a}*{b} -> {got} != {expect}"


@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_commutativity_and_distributivity(p, m, prim):
    gf = GF(p, m, prim)
    q = p ** m
    rng = random.Random(1337)

    for _ in range(min(200, q ** 2)):  # cap iterations for very small fields we hit a lot anyway
        a, b, c = rng.randrange(q), rng.randrange(q), rng.randrange(q)

        # Commutativity
        assert gf.add(a, b) == gf.add(b, a), "Addition not commutative"
        assert gf.mul(a, b) == gf.mul(b, a), "Multiplication not commutative"

        # Distributivity
        left = gf.mul(a, gf.add(b, c))
        right = gf.add(gf.mul(a, b), gf.mul(a, c))
        assert left == right, f"Distributivity failed in GF({p}^{m})"


@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_zero_annihilates_and_no_div_by_zero_surprises(p, m, prim):
    gf = GF(p, m, prim)
    q = p ** m
    for a in range(q):
        assert gf.mul(a, 0) == 0
        assert gf.mul(0, a) == 0


@pytest.mark.parametrize("p,m,prim", FIELDS)
def test_multiplicative_cyclic_nonzero_group_size(p, m, prim):
    """
    Nonzero elements should form a group of size q-1.
    We don't assert primitivity of 'x' here, just that repeated
    multiplication by a random nonzero cycles within the field.
    """
    gf = GF(p, m, prim)
    q = p ** m
    if q == 2:  # trivial tiny field
        return

    rng = random.Random(4242)
    a = rng.randrange(1, q)
    seen = set()
    x = 1
    for _ in range(q):  # at most q steps to cycle
        x = gf.mul(x, a)
        seen.add(x)
        if x == 1:
            break
    # We must have cycled; the order divides q-1.
    assert 1 in seen
    assert len(seen) >= 1 and len(seen) <= (q - 1)
