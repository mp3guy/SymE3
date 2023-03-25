import pytest, sys

sys.path.append('..')

from sympy import symbols, Matrix
from SymE3.core import PointH, Pixel, LieGroup, LieAlgebra, CustomFunction, TotalFunction, dehom, exp

def test_bundle_adjustment():
    x_w = PointH("{x_w}")
    x_i = Pixel("{x_i}")
    That_cw = LieGroup("{\\hat{T}_{cw}}")
    d = LieAlgebra("{\\delta}")
    f_x, f_y, c_x, c_y = symbols("f_x f_y c_x c_y")

    def proj(p):
        p_ray = p / p[2, 0]
        return Matrix([[f_x,   0, c_x],
                       [  0, f_y, c_y]]) * p_ray

    Pi = CustomFunction("Pi", proj, 3, 2)

    e = x_i - Pi(dehom(exp(d) * That_cw * x_w))

    f = TotalFunction(e)
    fe = f.as_explicit()
    df_dd = f.diff(d, dehom(x_w), f_x, f_y, c_x, c_y)
