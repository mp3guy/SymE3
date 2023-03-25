import pytest, sys

sys.path.append('..')

from SymE3.core import PointH, TotalFunction

def test_addition():
    a = PointH("a")
    b = PointH("b")
    e = TotalFunction(a + b).as_explicit().tomatrix()
    assert e[3] == 1

def test_subtraction():
    a = PointH("a")
    b = PointH("b")
    e = TotalFunction(a - b).as_explicit().tomatrix()
    assert e[3] == 1

def test_rmul():
    a = PointH("a")
    e = TotalFunction(2 * a).as_explicit().tomatrix()
    assert e[3] == 1

def test_mul():
    a = PointH("a")
    e = TotalFunction(a * 2).as_explicit().tomatrix()
    assert e[3] == 1
