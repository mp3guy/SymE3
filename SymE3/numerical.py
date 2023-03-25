import random

from sympy import MutableDenseMatrix, Symbol, Matrix, Float, eye, sin, cos
from sophus.se3 import Se3
from .detail import _Explicit, _MatrixSym

values = {}

def _resetValues():
    if len(values) == 0:
        random.seed(0)

def _realVal(s):
    if s in values:
        return values[s]
    else:
        value = random.uniform(-1, 1)
        values[s] = value
        return value

def _getRealMatValue(sym, idx):
    matrix = Matrix(_MatrixSym(sym.name, sym.rows, sym.cols))
    row = int(idx) // int(sym.cols)
    col = int(idx) % int(sym.cols)
    return matrix[row, col], _realVal(matrix[row, col])

def _subAndEvalReal(expression):
    if isinstance(expression, _Explicit):
        expression = expression.tomatrix()

    sub = {}

    # Recursively substitute placeholder values
    def recursiveSub(subExpr):
        if isinstance(subExpr, Symbol):
            sub[subExpr] = _realVal(subExpr)
        elif isinstance(subExpr, Matrix):
            for subSym in subExpr:
                recursiveSub(subSym)
        else:
            for subSym in subExpr.args:
                recursiveSub(subSym)

    recursiveSub(expression)

    # If we have symbols left, they are symbolic functions and we should store their derivatives. 
    substituted = expression.evalf(subs=sub)
    substituted = substituted.subs(sub)

    funcValues = {}

    def recursiveEval(subExpr):
        # We have found a symbolic function, manually evaluate a placeholder continuous function
        # and set that as the value of this function at the numerical point
        if isinstance(subExpr, Float):
            return subExpr

        if hasattr(subExpr, "inRows") and hasattr(subExpr, "outRows"):
            assert subExpr.inRows == len(subExpr.args)

            # Ensure all parameters have a value
            arguments = list(subExpr.args)

            for i in range(len(arguments)):
                if not isinstance(arguments[i], Float):
                    arguments[i] = recursiveEval(arguments[i])

            value = Float(0)

            for arg in arguments:
                value = value + sin(arg)

            # The partial derivative of the func w.r.t. any param is just cos(param), nice!
            for i in range(subExpr.inRows):
                partialSym = subExpr.fdiff(i + 1)
                values[partialSym] = cos(arguments[i])

            funcValues[subExpr] = value

            return value
        else:
            for subSym in subExpr.args:
                recursiveEval(subSym)

    if isinstance(substituted, MutableDenseMatrix):
        for elem in substituted:
            recursiveEval(elem)
    else:
        recursiveEval(substituted)

    substituted = substituted.subs(funcValues)

    if isinstance(substituted, Float):
        return Matrix([[substituted]])

    return substituted

def _exp(v, perturb):
    mat = Se3.exp(v.as_mutable()).matrix()

    # Work around singularity
    if perturb[3, 0] == 0 and perturb[4, 0] == 0 and perturb[5, 0] == 0:
        mat = eye(4)
        mat[0:3, 3] = perturb[0:3, 0]

    assert v.rows == 6
    assert v.shape == perturb.shape

    sub = {}

    for i in range(6):
        sub[v[i, 0]] = perturb[i, 0]

    return mat.evalf(subs=sub, strict=True)
