# %%
import importlib.readers
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import sys

sys.path.insert(1, "../")

import importlib

import sgGWR
from sgGWR import models, optimizers
from sgGWR.models import Ridge, OLS

# %%
# %load_ext autoreload
# %autoreload 2

# %%
n = 200
p = 10
rngkey = random.PRNGKey(123)
eps = 1.0

key, rngkey = random.split(rngkey)
beta = random.normal(key, shape=(p, 1))
key, rngkey = random.split(rngkey)
X = jnp.concatenate([jnp.ones((n, 1)), random.normal(key, (n, p - 1))], axis=1)
key, rngkey = random.split(rngkey)
y = X @ beta + eps * random.normal(key, shape=(n, 1))
# %%
ridge = Ridge(y, X, penalty=1e-2)
ridge.set_betas_inner()
ridge.get_beta().shape

beta_ = jnp.linalg.solve(
    ridge.X.T @ ridge.X + ridge.penalty * jnp.eye(p), ridge.X.T @ ridge.y
)

plt.bar(jnp.arange(p), beta.flatten(), label="true", width=0.3)
plt.bar(jnp.arange(p) + 0.3, beta_.flatten(), label="direct", width=0.3)
plt.bar(jnp.arange(p) + 0.3 * 2, ridge.beta.flatten(), label="svd", width=0.3)
plt.legend()
plt.show()


# %%
def direct_loocv(penalty):
    loss = 0.0
    for i in range(n):
        m = Ridge(
            y=jnp.delete(y, i, axis=0), X=jnp.delete(X, i, axis=0), penalty=penalty
        )
        b = m.get_beta()
        loss += jnp.square(y[i] - X[i] @ b)

    return loss / n


print(direct_loocv(ridge.penalty), ridge.loocv_loss())

# %%
hat = X @ jnp.linalg.inv(X.T @ X + ridge.penalty + jnp.eye(p)) @ X.T
# hat @ y - X @ ridge.beta
jnp.allclose(hat, ridge._hat(ridge.penalty), atol=1e-3)
# %%
ols = OLS(y, X)
ols.set_betas_inner()
# %%
penalties = jnp.logspace(-2, 1, endpoint=True)
plt.plot(penalties, [ridge.loocv_loss(p) for p in penalties], label="efficient")
# plt.plot(penalties, [direct_loocv(p) for p in penalties], label="direct")
plt.axhline(ols.loocv_loss(), c="k", label="OLS baseline")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("penalty")
plt.ylabel("loocv")
plt.show()
# %%
penalties = jnp.logspace(-2, 1, endpoint=True)
plt.plot(penalties, [ridge.AICc(p) for p in penalties], label="Ridge")
plt.axhline(ols.AICc(), c="k", label="OLS baseline")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("penalty")
plt.ylabel("AICc")
plt.show()

# %%
plt.plot(penalties, [ridge.loocv_loss(p) for p in penalties], label="efficient")
plt.axhline(ols.loocv_loss(), c="k", label="OLS baseline")
ridge.fit(aicc=False)
ridge.set_betas_inner()
plt.scatter(ridge.penalty, ridge.loocv_loss(), marker="x", c="red", label="optimized")
plt.legend()
plt.title("penalty={}".format(ridge.penalty))
plt.xscale("log")
# plt.yscale("log")
plt.xlabel("penalty")
plt.ylabel("loocv")
plt.show()

# %%
plt.plot(penalties, [ridge.AICc(p) for p in penalties], label="efficient")
plt.axhline(ols.AICc(), c="k", label="OLS baseline")
ridge.fit(aicc=True)

ridge.set_betas_inner()
plt.scatter(ridge.penalty, ridge.AICc(), marker="x", c="red", label="optimized")
plt.legend()
plt.title("penalty={}".format(ridge.penalty))
plt.xscale("log")
# plt.yscale("log")
plt.xlabel("penalty")
plt.ylabel("AICc")
plt.show()

# %%
