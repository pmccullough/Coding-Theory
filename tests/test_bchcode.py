import pytest
from codes.gf import GF
from codes.bch import BCHCode  # adjust if BCHCode is in a different file


# ---------- Fixtures ----------

@pytest.fixture
def gf8():
    # GF(2^3) with primitive polynomial x^3 + x + 1
    return GF(2, 3, [1, 0, 1, 1])

@pytest.fixture
def bch_7_4_1():
    # Narrow-sense BCH(7,4,1) = (n=7, t=1)
    return BCHCode(2, 3, [1, 0, 1, 1], t=1)

@pytest.fixture
def bch_15_7_2():
    # Narrow-sense BCH(15,7,2) over GF(2^4) with primitive poly x^4 + x + 1
    return BCHCode(2, 4, [1, 0, 0, 1, 1], t=2)


# ---------- GF tests ----------

def test_gf_add(gf8):
    a = [0, 1, 0]  # x
    b = [1, 0, 0]  # x^2
    result = gf8.add(a, b)
    assert gf8.coeffs(result) == [1, 1, 0]

def test_gf_mul(gf8):
    a = [0, 1, 0]  # x
    b = [1, 0, 0]  # x^2
    result = gf8.mul(a, b)
    # x * x^2 = x^3 = x + 1 in GF(2^3)
    assert gf8.coeffs(result) == [0, 1, 1]


# ---------- BCHCode helper tests ----------

def test_alpha_pow(bch_7_4_1):
    # α^0 = 1
    assert bch_7_4_1._alpha_pow(0) == [1, 0, 0]
    # α^1 = x
    assert bch_7_4_1._alpha_pow(1) == [0, 1, 0]

def test_cyclotomic_coset(bch_7_4_1):
    coset = bch_7_4_1._cyclotomic_coset(1)
    # Known coset in GF(2^3) modulo 7: {1,2,4}
    assert set(coset) == {1, 2, 4}

def test_minimal_polynomial_degree(bch_7_4_1):
    mpoly = bch_7_4_1._minimal_polynomial(1)
    # Degree should equal coset size (here: 3)
    assert len(mpoly) - 1 == len(bch_7_4_1._cyclotomic_coset(1))

def test_poly_mul():
    bch = BCHCode(2, 3, [1, 0, 1, 1], t=1)
    result = bch._poly_mul([1, 1], [1, 1])  # (x+1)*(x+1) over GF(2) = x^2 + 1
    assert result == [1, 0, 1]


# ---------- BCHCode integration tests ----------

def test_generator_polynomial_bch_7_4_1(bch_7_4_1):
    # Known generator polynomial for BCH(7,4,1) over GF(2) is x^3 + x + 1
    expected = [1, 0, 1, 1]
    assert bch_7_4_1.g == expected

def test_encode_bch_7_4_1(bch_7_4_1):
    message = [1, 0, 1, 1]
    codeword = bch_7_4_1.encode(message)
    # Codeword length should be n = 7
    assert len(codeword) == bch_7_4_1.n
    # Codeword should be divisible by g(x) in GF(2)
    remainder = poly_div(codeword, bch_7_4_1.g, p=2)[1]
    assert all(r == 0 for r in remainder)

def test_generator_polynomial_bch_15_7_2(bch_15_7_2):
    # Known generator polynomial for BCH(15,7,2) is degree 8
    # One common form: x^8 + x^7 + x^6 + x^4 + x^2 + x + 1
    expected_degree = 8
    assert len(bch_15_7_2.g) - 1 == expected_degree

def test_encode_bch_15_7_2(bch_15_7_2):
    message = [1, 0, 0, 1, 1, 0, 1]  # arbitrary 7-bit message
    codeword = bch_15_7_2.encode(message)
    # Codeword length should be n = 15
    assert len(codeword) == bch_15_7_2.n
    # Codeword should be divisible by g(x) in GF(2)
    remainder = poly_div(codeword, bch_15_7_2.g, p=2)[1]
    assert all(r == 0 for r in remainder)


# ---------- Utility for integration checks ----------

def poly_div(dividend, divisor, p=2):
    """Polynomial long division over GF(p), returns (quotient, remainder)."""
    dividend = dividend.copy()
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    for i in range(len(dividend) - len(divisor) + 1):
        coeff = dividend[i]
        if coeff != 0:
            quotient[i] = coeff
            for j in range(len(divisor)):
                dividend[i + j] = (dividend[i + j] - coeff * divisor[j]) % p
    remainder = dividend[-(len(divisor)-1):]
    return quotient, remainder
