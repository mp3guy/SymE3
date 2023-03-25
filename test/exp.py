import pytest, sys

sys.path.append('..')

from sympy import Array
from SymE3.core import exp, LieAlgebra, TotalFunction

def test_exp():
    d = LieAlgebra("{\\delta}")
    f = TotalFunction(exp(d))

    expected = Array([[[[0, 0, 0, 1], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0]]], 
                      [[[0, 0, 0, 0], 
                        [0, 0, 0, 1], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0]]], 
                      [[[0, 0, 0, 0], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 1], 
                        [0, 0, 0, 0]]], 
                      [[[0, 0, 0, 0], 
                        [0, 0, -1, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 0, 0]]], 
                      [[[0, 0, 1, 0], 
                        [0, 0, 0, 0], 
                        [-1, 0, 0, 0], 
                        [0, 0, 0, 0]]], 
                      [[[0, -1, 0, 0], 
                        [1, 0, 0, 0], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0]]]])

    result = f.diff(d)

    assert expected == result
