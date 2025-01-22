import ot
import ot.stochastic
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def compute_ot_dual(
    xs: np.ndarray,
    xt: np.ndarray,
    reg: float = 0.5,
    n_iter: int = 200,
    lr: float = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Computes the dual variables and transport plan for entropic regularized OT using stochastic optimization.

  Args:
    xs: Source samples (np.ndarray, shape (n_source, dim))
    xt: Target samples (np.ndarray, shape (n_target, dim))
    reg: Regularization parameter (float)
    n_iter: Number of iterations (int)
    lr: Learning rate for Adam optimizer (float)
    verbose: If True, prints the loss every 10 iterations (bool)

  Returns:
    u: Dual variable for source samples (np.ndarray, shape (n_source,))
    v: Dual variable for target samples (np.ndarray, shape (n_target,))
    Ge: Transport plan (np.ndarray, shape (n_source, n_target))
  """
  xs = torch.tensor(xs, dtype=torch.float32)
  xt = torch.tensor(xt, dtype=torch.float32)

  u = torch.randn(len(xs), requires_grad=True)
  v = torch.randn(len(xt), requires_grad=True)

  optimizer = torch.optim.Adam([u, v], lr=lr)

  losses = []

  for i in range(n_iter):
    # minus because we maximize the dual loss
    loss = -ot.stochastic.loss_dual_entropic(u, v, xs, xt, reg=reg)
    losses.append(float(loss.detach()))

    if verbose and i % 10 == 0:
        print("Iter: {:3d}, loss={}".format(i, losses[-1]))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  Ge = ot.stochastic.plan_dual_entropic(u, v, xs, xt, reg=reg)

  if verbose:
    plt.figure()
    plt.plot(losses)
    plt.grid()
    plt.title("Dual objective (negative)")
    plt.xlabel("Iterations")
    plt.show()

  return u.detach().numpy(), v.detach().numpy(), Ge