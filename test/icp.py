import pytest, sys

sys.path.append('..')

from sympy import Matrix
from SymE3.core import exp, LieAlgebra, NormalH, PointH, LieGroup, TotalFunction
from SymE3.detail import _MatrixSym

def test_point_plane_icp():
    n_ri = NormalH("{n_{r_i}}")
    r_i = PointH("{r_i}")
    l_i = PointH("{l_i}")
    rhat_i = PointH("{\\hat{r}_i}")
    That_rl = LieGroup("{\\hat{T}_{rl}}")
    d = LieAlgebra("{\\delta}")

    e = n_ri.transpose() * ((exp(d) * That_rl * l_i) - r_i)
    e = e.subs(That_rl * l_i, rhat_i)

    f = TotalFunction(e)
    df_dd = f.diff(d)

    # Compare against ground truth
    nr = Matrix(_MatrixSym(n_ri.name, 3, 1))
    r = Matrix(_MatrixSym(r_i.name, 3, 1))
    rh = Matrix(_MatrixSym(rhat_i.name, 3, 1))

    c = nr.transpose() * (rh - r)
    fe = f.as_explicit()
    assert c == fe.tomatrix()

    cp = rh.cross(nr).transpose()
    jc = nr.transpose()
    jc = jc.col_insert(3, cp)
    assert jc == df_dd
