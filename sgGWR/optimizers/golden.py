"""
Golden Section Search for adaptive bandwidth 
"""

import numpy as np

from tqdm.auto import tqdm
from .. import models

__all__ = ["golden_section"]


class golden_section(object):
    def __init__(self) -> None:
        super().__init__()

    def run(
        self, model, maxiter=1000, bracket=None, tol=1e-5, aicc=False, verbose=True
    ):
        assert isinstance(model, models.GWR)

        if bracket is None:

            a, c = model.D + 1, model.N
        else:
            a, c = bracket
            assert (type(a) is int) and (type(c) is int)
            if a > c:
                a, c = c, a

        if aicc:

            def f(x):
                return model.AICc(np.array([x]))

        else:

            def f(x):
                return model.loocv_loss(np.array([x]))
                # return self._loocv(x, model)

        fa, fc = f(a), f(c)

        # check interval
        if a == c:
            opt = a
            self.loss = [fa]
        elif a + 1 == c:
            if fa < fc:
                self.loss = [fa]
                opt = a
            else:
                self.loss = [fc]
                opt = c

        delta = (1 + np.sqrt(5)) / 2 - 1

        # make triplet [a, b, c]
        b = self._propose_new(a, c, delta)
        fb = f(b)

        self.loss = [float(fb)]
        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for i in range(maxiter):
                pbar.update(1)
                pbar.set_description(
                    "loss = {:.2f}, trpilet={}".format(self.loss[-1], (a, b, c))
                )

                # fmt: off
                # do not format code between fmt: off and fmt: on
                if b - a >= c - b:
                    # propose new point in [a, b]
                    d = self._propose_new(a, b, delta)
                    fd = f(d)

                    # compare f(d) with center value f(b) and get new triplet
                    if fd < fb:
                        # next triplet is [a, d, b]
                        a, b, c = a, d, b
                        fa, fb, fc = fa, fd, fb
                    else:
                        # next triplet is [d, b, c]
                        a, b, c, = d, b, c
                        fa, fb, fc, = fd, fb, fc
                else:
                    # propose new point in [b, c]
                    d = self._propose_new(b, c, delta)
                    fd = f(d)

                    # compare f(d) with center value f(b) and get new triplet
                    if fd < fb:
                        # next triplet is [b, d, c]
                        a, b, c, =  b, d, c
                        fa, fb, fc, = fb, fd, fc
                    else:
                        # next triplet is [a, b, d]
                        a, b, c = a, b, d
                        fa, fb, fc = fa, fb, fd
                # fmt: on

                if not (a < b and b < c):
                    raise ValueError(f"the triplet {[a, b ,c]} is not in order.")

                # check convergence
                idx = np.argmin([fa, fb, fc])
                opt = [a, b, c][idx]
                self.loss.append(float([fa, fb, fc][idx]))
                if max(fa, fb, fc) - min(fa, fb, fc) < tol or a + 2 >= c:
                    pbar.set_description(
                        "loss = {:.2f}, trpilet={}".format(self.loss[-1], (a, b, c))
                    )
                    break
            else:
                print(
                    "Caution: golden section search reached maximum number of iteration. maxiter may not enough large."
                )

        model.kernel.params = [opt]
        return self.loss

    def _propose_new(self, a, c, delta):
        b = delta * a + (1 - delta) * c
        b_floor, b_ceil = np.floor(b), np.ceil(b)

        if b_floor != a and b_ceil != c:
            b = np.round(b)
        elif b_floor == a:
            b = b_ceil
        elif b_ceil == c:
            b = b_floor

        return int(b)
