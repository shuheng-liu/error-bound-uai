import numpy as np
from scipy.integrate import cumulative_trapezoid


def cumulative_trapezoid_upper(y, x, initial=0):
    z1 = np.concatenate([y[1:], [y[-1]]])
    z2 = np.concatenate([[y[0]], y[:-1]])
    assert z1.shape == z2.shape == y.shape
    y = np.maximum(y, z1)
    y = np.maximum(y, z2)
    return cumulative_trapezoid(y, x, initial=initial)


class OperatorI:
    def __init__(self, lmd, *, integral_multiplicity=1, ceil=True):
        assert isinstance(lmd, (int, float, complex))

        self.lmd = lmd
        self._integral_multiplicity = integral_multiplicity
        self.t = None
        self.ceil = ceil

    def bind_domain(self, t=None):
        self.t = t
        return self

    def __pow__(self, other):
        assert isinstance(other, int)

        new = self.__class__(
            self.lmd,
            integral_multiplicity=self._integral_multiplicity * other
        )
        new.bind_domain(self.t)
        return new

    def __call__(self, psi, t=None):
        if t is None:
            t = self.t

        assert (psi >= 0).all()
        assert isinstance(psi, np.ndarray) and isinstance(t, np.ndarray)
        assert psi.shape == t.shape and len(t.shape) == 1

        integral = psi * np.exp(-self.lmd * t)
        for _ in range(self._integral_multiplicity):
            if self.ceil:
                integral = cumulative_trapezoid_upper(integral, t, initial=0)
            else:
                integral = cumulative_trapezoid(integral, t, initial=0)
        return integral * np.exp(self.lmd * t)


class OperatorIStable(OperatorI):
    def __init__(self, lmd, *, integral_multiplicity=1, ceil=True):
        assert (isinstance(lmd, complex) and lmd.real <= 0) or lmd <= 0
        super().__init__(lmd, integral_multiplicity=integral_multiplicity, ceil=ceil)
