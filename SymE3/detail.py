from enum import Enum

from sympy import symbols, tensorproduct, tensorcontraction, transpose, Function
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

class _Type(Enum):
    POINTH, NORMALH = range(2)

def _MatrixSym(prefix, R, C):
    M = []
    for r in range (0, R):
        Row = []
        for c in range (0, C):
            name = '{{{0}_{{{1}{2}}}}}'.format(prefix, r if (R > 1) else '', c if (C > 1) else '' )
            Row.append(symbols(name))
        M.append(Row)
    return M

def _PointH(name):
    vector = _MatrixSym(name, 3, 1)
    vector.append([1])
    result = _Explicit(vector)
    result.type = _Type.POINTH
    return result

def _NormalH(name):
    vector = _MatrixSym(name, 3, 1)
    vector.append([0])
    result = _Explicit(vector)
    result.type = _Type.NORMALH
    return result

def _Point(name):
    return _Explicit(_MatrixSym(name, 3, 1))

def _Normal(name):
    return _Explicit(_MatrixSym(name, 3, 1))

def _Pixel(name):
    return _Explicit(_MatrixSym(name, 2, 1))

def _Plane(name):
    return _Explicit(_MatrixSym(name, 4, 1))

def _Matrix3(name):
    return _Explicit(_MatrixSym(name, 3, 3))

def _LieAlgebra(name):
    return _Explicit(_MatrixSym(name, 6, 1))

def _LieGroup(name):
    syms = _MatrixSym(name, 4, 4)
    syms[3][0] = 0
    syms[3][1] = 0
    syms[3][2] = 0
    syms[3][3] = 1
    return _Explicit(syms)

def _ExponentialMap(name):
    def new(self, d):
        assert d.shape == (6, 1)
        return _Explicit([[       1, -d[5, 0],  d[4, 0], d[0, 0]], 
                          [ d[5, 0],        1, -d[3, 0], d[1, 0]], 
                          [-d[4, 0],  d[3, 0],        1, d[2, 0]], 
                          [       0,        0,        0,      1]])

    exp = type(f"exp_{name}", (Function, ), {"__new__": new})

    return exp;

class _Explicit(ImmutableDenseNDimArray):
    def diff(self, *args, **kwargs):

        if isinstance(args[0], list):
            for symbol in args[0]:
                if (isinstance(symbol, list) and (symbol[0] == 0 or symbol[0] == 1)) \
                        or (symbol == 0 or symbol == 1):
                    print("Scalar detected in diff input, did you forget to dehom?")

        result = super().diff(*args, **kwargs)

        if hasattr(args[0], "rank"):
            # Catch the edge case where matrix differentiation doesn't work for some reason
            if result.rank() != 4 and args[0].rank() == 2 and self.rank() == 2:
                result = self.tomatrix().diff(args[0].tomatrix())

        if result.rank() == 4 and result.shape[1] == 1 and result.shape[3] == 1:
            result = tensorcontraction(result, (1, 3))

        return _Explicit(result)

    def __new__(cls, iterable, shape=None, **kwargs):
        self = super().__new__(cls, iterable, shape, **kwargs)
        self.type = None
        return self

    def __mul__(self, other):
        result = _Explicit(tensorcontraction(tensorproduct(self, other), (1, 2)))

        if hasattr(other, "type") and result.shape == (4, 1):
            result.type = other.type

        return result

    def __rmul__(self, other):
        result = _Explicit(super().__rmul__(other))

        if hasattr(self, "type"):
            if self.type == _Type.POINTH and result.shape == (4, 1):
                explicit = result.tomatrix()
                explicit[3, 0] = 1
                result = _Explicit(explicit)
                result.type = _Type.POINTH

        return result

    def __add__(self, other):
        result = _Explicit(super().__add__(other))

        if hasattr(self, "type") and hasattr(other, "type"):
            assert self.type == other.type
            
            if self.type == _Type.POINTH:
                explicit = result.tomatrix()
                explicit[3, 0] = 1
                result = _Explicit(explicit)
                result.type = _Type.POINTH

        return result

    def transpose(self):
        return _Explicit(transpose(self))

    def inverse(self):
        return _Explicit(self.tomatrix().inv())

