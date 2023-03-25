import pytest, sys

sys.path.append('..')

from sympy import symbols, eye, Matrix
from SymE3.core import Plane, LieGroup, PointH, Pixel, LieAlgebra, CustomFunction, TotalFunction, dehom, exp

def test_mirrors():
    T_cw = LieGroup("{T_{cw}}")
    T_ct = LieGroup("{\hat{T}_{ct}}")
    p_t = PointH("{p_t}")
    phat_c = PointH("{\hat{p}_{c}}")
    p_c = Pixel("{p_c}")
    N_w = Plane("{N_w}")
    d = LieAlgebra("{\\delta}")

    def proj(p):
        p_ray = p / p[2, 0]
        f_x, f_y, c_x, c_y = symbols("f_x f_y c_x c_y")
        
        return Matrix([[f_x,   0, c_x],
                       [  0, f_y, c_y]]) * p_ray

    Pi = CustomFunction("Pi", proj, 3, 2)

    def sym(n):
        n_hat = n[0:3, :]
        S = eye(4)
        S[0:3, 0:3] = eye(3) - (2 * (n_hat * n_hat.transpose()))
        S[0:3, 3] = 2 * n[3] * n_hat
        return S
        
    S = CustomFunction("S", sym, 4, 4, 1, 4)

    e = Pi(dehom(T_cw * S(N_w) * T_cw.inverse() * exp(d) * T_ct * p_t)) - p_c
    e = e.subs(T_ct * p_t, phat_c)
    f = TotalFunction(e)

    fe = f.as_explicit()
    df_dd = f.diff(d, N_w)


