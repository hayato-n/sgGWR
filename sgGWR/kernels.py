try:
    from jax import numpy as jnp

    def _mask_loocv(w, i):
        if i is None:
            return w
        else:
            return w.at[i].set(0.0)


except:
    import numpy as jnp

    def _mask_loocv(w, i):
        if i is None:
            return w
        else:
            w[i] = 0.0
            return w

    def vmap(f):
        def vf(idx):
            return jnp.stack(list(map(f, idx)), axis=0)

        return vf


from scipy.spatial import KDTree

__all__ = [
    "Gaussian",
    "Exponential",
    "Epanechnikov",
    "Triangular",
    "Biweight",
    "LinearMultiscale",
    "stGaussian",
    "stExponential",
    "stEpanechnikov",
    "stTriangular",
    "stBiweight",
]


class _baseKernel(object):
    def __init__(self, params):
        self.params = params

    def k(self, x1, x2, params):
        raise NotImplementedError()

    def dk(self, x1, x2, params):
        raise NotImplementedError()

    def __call__(self, x1, x2=None, params=None, loocv=None):
        return self.forward(x1, x2, params, loocv)

    def forward(self, x1, x2=None, params=None, loocv=None):
        if x2 is None:
            x2 = x1.reshape(1, -1)
        if params is None:
            params = self.params

        return _mask_loocv(self.k(x1, x2, params), loocv)
        # if loocv:
        #     return jnp.where(jnp.all(x1 == x2, axis=1), 0, self.k(x1, x2, params))
        # else:
        #     return self.k(x1, x2, params)

    def grad(self, x1, x2=None, params=None, loocv=None):
        if x2 is None:
            x2 = x1.reshape(1, -1)
        if params is None:
            params = self.params

        return _mask_loocv(self.dk(x1, x2, params), loocv)
        # if loocv:
        #     return jnp.where(
        #         jnp.all(x1 == x2, axis=1)[:, None], 0, self.dk(x1, x2, params)
        #     )
        # else:
        #     return self.dk(x1, x2, params)


class _scaledKernel(_baseKernel):
    def _dist(self, x1, x2, params):
        d = jnp.linalg.norm(x1 - x2, axis=-1)
        return params[0] * d, d

    def _scaledk(self, d):
        raise NotImplementedError()

    def _scaleddk(self, d):
        raise NotImplementedError()

    def k(self, x1, x2, params):
        scaled, _ = self._dist(x1, x2, params)
        return self._scaledk(scaled)

    def dk(self, x1, x2, params):
        scaled, d = self._dist(x1, x2, params)
        return (self._scaleddk(scaled) * d).reshape(-1, 1)

    def _scaled_k_inv(self, k):
        raise NotImplementedError()

    def init_param(self, sites, idx=None):
        if idx is None:
            idx = jnp.arange(len(sites), dtype=int)

        # apply map to avoid making N*N matrix
        med_dists = list(
            map(lambda i: jnp.median(self._dist(sites[i], sites, [1.0])[1]), idx)
        )
        d_median = jnp.median(jnp.array(med_dists))

        # search param that satisfy 0.5 * k(param0 * d_median) = k(0)
        # -> param0 = k^{-1}(0.5 * k(0)) / d_median
        k0 = self._scaledk(0.0)
        self.params = [self._scaled_k_inv(0.5 * k0) / d_median]


class Gaussian(_scaledKernel):
    def _scaledk(self, d):
        return jnp.exp(-jnp.square(d))

    def _scaleddk(self, d):
        return -2 * d * self._scaledk(d)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(-jnp.log(k))


class Exponential(_scaledKernel):
    def _scaledk(self, d):
        return jnp.exp(-d)

    def _scaleddk(self, d):
        return -self._scaledk(d)

    def _scaled_k_inv(self, k):
        return -jnp.log(k)


class Epanechnikov(_scaledKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, 1 - jnp.square(d), 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -2 * d, 0)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(1 - k)


class Triangular(_scaledKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, 1 - d, 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -1, 0)

    def _scaled_k_inv(self, k):
        return 1 - k


class Biweight(_scaledKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, jnp.square(1 - jnp.square(d)), 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -4 * d * (1 - jnp.square(d)), 0)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(1 - jnp.sqrt(k))


class _KDTreeKernel(_baseKernel):
    def __init__(self, sites, params):
        super().__init__(params)
        self._kdtree = KDTree(sites)

    def _knn(self, sites, k=1):
        dist, idx = self._kdtree.query(sites, k=k)
        return dist, idx


class LinearMultiscale(_KDTreeKernel):
    def __init__(
        self,
        sites,
        params=jnp.array([0.01, 1]) ** 2,
        base_kernel=None,
        n_poly=4,
        n_neighbour=100,
    ):
        # default settings are the same to that of 'scgwr' package in R
        # see:  https://github.com/cran/scgwr/blob/master/R/scgwr.R
        super().__init__(sites, params)
        assert n_poly >= 1
        self.n_poly = int(n_poly)
        assert n_neighbour >= 1
        self.n_neighbour = int(n_neighbour)

        self.sites = sites

        if base_kernel is None:

            knn_dist, _ = self._knn(sites, k=self.n_neighbour)
            ban0 = jnp.median(
                knn_dist[:, min(50 + 1, self.n_neighbour + 1)]
            ) / jnp.sqrt(3)
            self.base_kernel = Gaussian([1 / ban0])
        else:
            self.base_kernel = base_kernel
        self.N = len(self.sites)

    def k(self, x1, x2, params):
        a = params[0]
        b = params[1:]
        w0 = self.base_kernel(x1, x2, loocv=None)
        # assume constantly decreasing kernel
        # thres = jnp.partition(w0, self.n_neighbour)[self.n_neighbour - 1] # jnp.partition is not implemented yet
        thres = jnp.sort(w0)[::-1][self.n_neighbour - 1]
        w0 = jnp.where(w0 <= thres, 0, w0)

        p = jnp.arange(1, self.n_poly + 1, dtype=int)
        if len(jnp.array(b)) == 1:
            b = b ** p
        Wp = w0[None] ** (4 / jnp.power(2, p))[:, None]
        return a + jnp.sum(b[:, None] * Wp, axis=0)

    def dk(self, x1, x2, params):
        raise NotImplementedError()


class _scaledSTKernel(_scaledKernel):
    """STkernel (space-time kernel) assumes three dimentional coordinates.
    First two coordinates are location (x,y) coordinates, whearas the last third coordinate is the temporal time-stamp t
    Space-time distance is determined as follows:
        [Space-time distance] = params[0] * [spatial distance] + params[1] * [temporal distance]
    """

    def _dist(self, x1, x2, params):
        vec = x1 - x2

        d = jnp.linalg.norm(vec[:, :2], axis=-1)
        t = jnp.abs(vec[:, 2])
        return params[0] * d + params[1] * t, d, t

    def k(self, x1, x2, params):
        scaled, _, _ = self._dist(x1, x2, params)
        return self._scaledk(scaled)

    def dk(self, x1, x2, params):
        scaled, d, t = self._dist(x1, x2, params)
        dkdx = self._scaleddk(scaled)
        return jnp.stack([dkdx * d, dkdx * t], axis=1)

    def _scaled_k_inv(self, k):
        raise NotImplementedError()

    def init_param(self, sites, idx=None):
        if idx is None:
            idx = jnp.arange(len(sites), dtype=int)

        # apply map to avoid making N*N matrix
        med_dists = []
        med_ts = []
        for i in idx:
            tmp = self._dist(sites[i], sites, [1.0, 1.0])
            med_dists.append(jnp.median(tmp[1]))
            med_ts.append(jnp.median(tmp[2]))

        d_median = jnp.median(jnp.array(med_dists))
        t_median = jnp.median(jnp.array(med_ts))

        dt_scale = d_median / t_median
        # search param that satisfy 0.5 * k(param0 * 2 * d_median) = k(0)
        # -> param0 = k^{-1}(0.5 * k(0)) / d_median / 2
        k0 = self._scaledk(0.0)
        param0 = self._scaled_k_inv(0.5 * k0) / d_median / 2

        # param0 * 2 * d_median = param0 * (d_median + dt_scale * t_median)
        # -> param1 = param0 * dt_scale
        param1 = param0 * dt_scale

        self.params = [param0, param1]


class stGaussian(_scaledSTKernel):
    def _scaledk(self, d):
        return jnp.exp(-jnp.square(d))

    def _scaleddk(self, d):
        return -2 * d * self._scaledk(d)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(-jnp.log(k))


class stExponential(_scaledSTKernel):
    def _scaledk(self, d):
        return jnp.exp(-d)

    def _scaleddk(self, d):
        return -self._scaledk(d)

    def _scaled_k_inv(self, k):
        return -jnp.log(k)


class stEpanechnikov(_scaledSTKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, 1 - jnp.square(d), 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -2 * d, 0)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(1 - k)


class stTriangular(_scaledSTKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, 1 - d, 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -1, 0)

    def _scaled_k_inv(self, k):
        return 1 - k


class stBiweight(_scaledSTKernel):
    def _scaledk(self, d):
        return jnp.where(d <= 1, jnp.square(1 - jnp.square(d)), 0)

    def _scaleddk(self, d):
        return jnp.where(d <= 1, -4 * d * (1 - jnp.square(d)), 0)

    def _scaled_k_inv(self, k):
        return jnp.sqrt(1 - jnp.sqrt(k))


class SpectralMixture(_baseKernel):
    """Spectral Mixture (SM) kernel for Gaussian Process

    inspired by

    Andrew Wilson, Ryan Adams
    Gaussian Process Kernels for Pattern Discovery and Extrapolation
    Proceedings of the 30th International Conference on Machine Learning, PMLR
    28(3):1067-1075, 2013.
    http://proceedings.mlr.press/v28/wilson13.html
    """

    def __init__(self, params):
        super().__init__(params)

        if len(self.params) % 3 != 0:
            raise ValueError("number of parameters must be (3 * number of mixture)")

        self.n_mixture = len(self.params) / 3

    def _dist(self, x1, x2):
        d = jnp.linalg.norm(x1 - x2, axis=-1)
        return d

    def _parse_params(self, params):
        mat = jnp.array(params).reshape((3, self.n_mixture))
        w, v, mu = mat[0], mat[1], mat[2]
        return w, v, mu

    def k(self, x1, x2, params):
        w, v, mu = self._parse_params(params)
        d = self._dist(x1, x2)

        e, c = self._get_ec(d, v, mu)
        return jnp.sum(w * e * c)

    def dk(self, x1, x2, params):
        w, v, mu = self._parse_params(params)
        d = self._dist(x1, x2)

        e, c = self._get_ec(d, v, mu)

        dw = e * c
        dv = w * c * e * (-2 * jnp.square(jnp.pi * d))
        dmu = w * e * (-2 * jnp.pi * d) * jnp.sin(2 * jnp.pi * d * mu)

        return jnp.concatenate([dw, dv, dw]).reshape(-1, 1)

    def _get_ec(self, d, v, mu):
        e = jnp.exp(-2 * jnp.square(jnp.pi * d) * v)
        c = jnp.cos(2 * jnp.pi * d * mu)

        return e, c

