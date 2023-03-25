import pytest, sys

sys.path.append('..')

from SymE3.core import exp, LieAlgebra, LieGroup, log

def test_log():
    T_i = LieGroup("{T_{i}}")
    deltaxi_i = LieAlgebra("{\\delta\\xi_{i}}")

    assert exp(log(T_i)) == T_i

    assert log(exp(deltaxi_i)) == deltaxi_i
