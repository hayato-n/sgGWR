# %%
import importlib.readers
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import sys

sys.path.insert(1, "../")

import importlib

import sgGWR
from sgGWR import models, kernels, optimizers
from sgGWR.models import GWR


# %%
importlib.reload(sgGWR)
importlib.reload(optimizers)
importlib.reload(kernels)

# %%
u0, v0 = jnp.arange(25), jnp.arange(25)
u, v = jnp.meshgrid(u0, v0)
u, v = u.flatten(), v.flatten()
N = len(u)
rngkey = random.PRNGKey(123)

beta = jnp.stack(
    [
        3 * jnp.ones(N),
        1 + (u + v) / 12,
        1 + (36 - jnp.square(6 - u / 2)) * (36 - jnp.square(6 - v / 2)) / 324,
    ]
).T
X = jnp.concatenate([jnp.ones((N, 1)), random.normal(rngkey, shape=(N, 2))], axis=1)
y = jnp.sum(X * beta, axis=1) + 0.5 * random.normal(rngkey, shape=(N,))
# %%
fig, axes = plt.subplots(1, 3, subplot_kw={"aspect": "equal"}, figsize=(10, 5))

for d in range(3):
    ct = axes[d].contourf(u0, v0, beta[:, d].reshape(len(u0), len(v0)))
    fig.colorbar(ct, ax=axes[d])

fig.tight_layout()
plt.show()

# %%
sites = jnp.stack([u, v]).T
kernel = kernels.AdaptiveKernel(params=[10], k_dist=kernels.Biweight([10]))
gwr = models.GWR(y, X, sites, kernel)

bws = range(gwr.D, gwr.N, 10)
plt.plot(bws, [gwr.loocv_loss([bw]) for bw in bws])
plt.show()

plt.plot(bws, [gwr.AICc([bw]) for bw in bws])
plt.show()

# %%
optim = optimizers.golden_section()
optim.run(gwr, maxiter=1000)
gwr.set_betas_inner()

plt.plot(optim.loss)
plt.title(f"bw={gwr.kernel.params[0]}")
plt.show()

# %%
plt.plot(bws, [gwr.loocv_loss([bw]) for bw in bws])
plt.scatter(
    gwr.kernel.params[0],
    gwr.loocv_loss([gwr.kernel.params[0]]),
    marker="x",
    color="red",
)
plt.show()

# %%
optim = optimizers.golden_section()
optim.run(gwr, maxiter=1000, aicc=True)
gwr.set_betas_inner()

plt.plot(optim.loss)
plt.title(f"bw={gwr.kernel.params[0]}")
plt.show()

# %%
plt.scatter(
    gwr.kernel.params[0], gwr.AICc([gwr.kernel.params[0]]), marker="x", color="red"
)
plt.plot(bws, [gwr.AICc([bw]) for bw in bws])
plt.show()

# %%
fig, axes = plt.subplots(1, 3, subplot_kw={"aspect": "equal"}, figsize=(10, 5))

for d in range(3):
    ct = axes[d].contourf(u0, v0, gwr.betas[:, d].reshape(len(u0), len(v0)))
    fig.colorbar(ct, ax=axes[d])

fig.tight_layout()
plt.show()

# %%
mgwr = models.MGWR(
    y=y, X=X, sites=sites, base_class=models.GWR, kernel=kernels.AdaptiveKernel([10])
)
optims = [optimizers.golden_section()] * 3
mgwr.backfitting(optimizers=optims, run_params={"aicc": True, "verbose": False})
print([k.params for k in mgwr.kernel])

plt.plot(mgwr.RSS)
plt.show()

fig, axes = plt.subplots(1, 3, subplot_kw={"aspect": "equal"}, figsize=(10, 5))

for d in range(3):
    ct = axes[d].contourf(u0, v0, mgwr.betas[:, d].reshape(len(u0), len(v0)))
    fig.colorbar(ct, ax=axes[d])

fig.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(3, 3, subplot_kw={"aspect": "equal"}, figsize=(10, 10))

for d in range(3):
    ct = axes[0][d].contourf(
        u0, v0, beta[:, d].reshape(len(u0), len(v0)), vmin=1.0, vmax=5.0
    )
    fig.colorbar(ct, ax=axes[0][d])
    ct = axes[1][d].contourf(
        u0, v0, gwr.betas[:, d].reshape(len(u0), len(v0)), vmin=1.0, vmax=5.0
    )
    fig.colorbar(ct, ax=axes[1][d])
    ct = axes[2][d].contourf(
        u0, v0, mgwr.betas[:, d].reshape(len(u0), len(v0)), vmin=1.0, vmax=5.0
    )
    fig.colorbar(ct, ax=axes[2][d])

fig.tight_layout()
plt.show()

# %%
gwr.kernel.params, [k.params for k in mgwr.kernel]

# %%
plt.axis("equal")
plt.scatter(mgwr.betas[:, 0], gwr.betas[:, 0], s=1, marker="x")
plt.axvline(3, color="k")
plt.axhline(3, color="k")
plt.xlabel("MGWR estimate")
plt.ylabel("GWR estimate")
plt.show()


# %%
# %load_ext autoreload
# %autoreload 2

# %%
