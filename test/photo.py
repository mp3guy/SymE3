import pytest, sys

sys.path.append('..')

from sympy import expand, symbols, Matrix, tensorcontraction
from SymE3.core import Pixel, PointH, LieGroup, LieAlgebra, CustomFunction, SymbolicFunction, dehom, TotalFunction, exp
from SymE3.detail import _MatrixSym

def test_photometric_alignment():
    x_i = Pixel("{x_i}")
    p_i = PointH("{p_i}")
    That_rl = LieGroup("{\\hat{T}_{rl}}")
    d = LieAlgebra("{\\delta}")
    phat_i = PointH("{\\hat{p}_i}")

    def proj(p):
        p_ray = p / p[2, 0]
        f_x, f_y, c_x, c_y = symbols("f_x f_y c_x c_y")
        
        return Matrix([[f_x,   0, c_x],
                        [  0, f_y, c_y]]) * p_ray

    Pi = CustomFunction("Pi", proj, 3, 2)
    I_r = SymbolicFunction("I_r", 2, 1)
    I_l = SymbolicFunction("I_l", 2, 1)

    e = I_r(Pi(dehom(exp(d) * That_rl * p_i))) - I_l(x_i)
    e = e.subs(That_rl * p_i, phat_i)
    f = TotalFunction(e)
    df_dd = f.diff(d)

    # Compare against ground truth
    ph = Matrix(_MatrixSym(phat_i.name, 3, 1))
    x = Matrix(_MatrixSym(x_i.name, 2, 1))
    pi = Pi.__explicit__()(ph).tomatrix()
    il = I_l.__explicit__()(x[0], x[1])
    ir = I_r.__explicit__()(pi[0], pi[1])

    fe = f.as_explicit()
    c = ir - il
    assert c.__str__() == fe.__str__()

    dpi_dph = tensorcontraction(pi.diff(ph), (1, 3)).transpose()
    dir_dpi = Matrix([ir.fdiff(1), ir.fdiff(2)]).transpose()
    gradpi = dir_dpi * dpi_dph
    cp = ph.cross(gradpi).transpose()
    jc = gradpi
    jc = jc.col_insert(3, cp)
    df_ddm = df_dd
    for i in range(6):
        jc[i] = expand(jc[i])
        df_ddm[i] = expand(df_ddm[i])
    assert jc == df_ddm
