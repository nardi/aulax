# Theory

This page derives the augmented Lagrangian method, shows how aulax runs it, and explains what each solver parameter changes. We start from the problem

$$
\min_y f(y) \quad \text{subject to} \quad h(y) = 0, \quad g(y) \le 0,
$$

where $y$ collects all variables, $h$ returns the equality residuals and $g$ returns the inequality residuals. For most of the derivation we look at equality constraints only, and we add the inequalities at the end.

## Penalty method

One general approach to a constrained problem is to look for an unconstrained problem with the same solution. The simplest candidate adds a penalty on constraint violations to the objective:

$$
\min_y f(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

For any finite penalty $\rho$, the minimizer of this function violates the constraints a little. The objective pulls $y$ away from the feasible set, and the penalty only pushes back in proportion to the violation. At a minimizer $y_\rho$, the gradient is zero, so

$$
\nabla f(y_\rho) + \rho \, \nabla h(y_\rho)^\top h(y_\rho) = 0.
$$

Compare this with the optimality condition of the constrained problem, where a multiplier vector $\lambda^*$ satisfies $\nabla f(y^*) + \nabla h(y^*)^\top \lambda^* = 0$. The two match when $\rho \, h(y_\rho) \approx \lambda^*$, so the violation is about $\lambda^* / \rho$. The only way to make it small is a large $\rho$. Unfortunately, a large $\rho$ also makes the Hessian badly conditioned, and every unconstrained minimiser gets slower or less accurate as a result.

## Augmented Lagrangian

The Lagrangian $f(y) + \lambda^\top h(y)$ has the right stationary point if we knew $\lambda^*$, but we don't, and the solution is a saddle point of it rather than a minimum. The augmented Lagrangian combines both approaches:

$$
L_\rho(y; \lambda) = f(y) + \lambda^\top h(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

We can read it either as the Lagrangian with an extra penalty on violations, or as the penalty function turned into a Lagrangian. Its gradient is

$$
\nabla_y L_\rho = \nabla f(y) + \nabla h(y)^\top \big( \lambda + \rho \, h(y) \big).
$$

At a minimizer of $L_\rho$, the vector $\lambda + \rho \, h(y)$ plays the role of the multipliers in the optimality condition. This gives the update rule

$$
\lambda \leftarrow \lambda + \rho \, h(y),
$$

and the method alternates between minimizing $L_\rho$ over $y$ and applying this update. If $\lambda$ is already equal to $\lambda^*$, the minimizer of $L_\rho$ is the exact solution for any $\rho$ large enough to make the problem locally convex. The penalty no longer has to grow without bound, it only has to be large enough.

## Shifted-penalty view

There is a second way to see why this works. Completing the square gives

$$
L_\rho(y; \lambda) = f(y) + \frac{\rho}{2} \left\lVert h(y) + \frac{\lambda}{\rho} \right\rVert^2 - \frac{\lVert \lambda \rVert^2}{2\rho}.
$$

The last term does not depend on $y$, so minimizing $L_\rho$ is the same as minimizing a penalty function whose target is shifted from $h = 0$ to $h = -\lambda / \rho$. The penalty method alone lands at $h \approx \lambda^* / \rho$. Shifting the target by exactly that amount in the other direction makes the minimizer land on $h = 0$. The multiplier update moves the shift until this happens. This view is worked out in more detail in [Augmented Lagrangian methods as modified penalty methods](https://nardilam.nl/articles/alm-penalty/).

## Inequality constraints

An inequality constraint $g_j(y) \le 0$ should only be penalized when it is violated, or close to it. With multipliers $\mu \ge 0$, the Powell-Hestenes-Rockafellar form of the augmented Lagrangian is

$$
L_\rho(y; \lambda, \mu) = f + \lambda^\top h + \frac{\rho}{2} \lVert h \rVert^2 + \frac{1}{2\rho} \left( \lVert \max(0, \mu + \rho g) \rVert^2 - \lVert \mu \rVert^2 \right),
$$

and the multiplier update becomes $\mu \leftarrow \max(0, \mu + \rho g)$. In the shifted-penalty view, the inequality term is $\frac{\rho}{2} \lVert \max(0, g + \mu / \rho) \rVert^2$ up to a constant. A constraint with a positive multiplier is penalized from $g = -\mu_j / \rho$ onwards, so its shifted target lies strictly inside the feasible region. A constraint that is comfortably satisfied has $\mu_j = 0$ and contributes nothing.

The $\max$ makes the second derivative of $L_\rho$ jump wherever $g_j = -\mu_j / \rho$. Quasi-Newton minimisers like BFGS build their Hessian approximation from gradient differences, and they lose accuracy when many constraints cross that boundary during an inner solve. Problems with many bound constraints need more inner steps for this reason.

## Solver loop

aulax minimizes each $L_\rho$ with an optimistix minimiser, which it calls one step at a time inside a single `jax.lax.while_loop`. Each loop iteration takes one inner step. When the inner minimiser reports that it has converged, the next iteration updates the multipliers and the penalty, and restarts the inner minimiser at the same point.

The update happens one iteration late, because an optimistix minimiser returns the evaluation of the point it started its step from. The solver needs $h$ and $g$ at the inner solution to update the multipliers, so it takes one more step from that point, uses the evaluation that comes back, and discards the step itself.

A single flat loop matters under `jax.vmap`. A nested loop, with an outer loop around a complete inner solve, would make every problem in a batch wait for the slowest inner solve before any of them could update its multipliers. In the flat loop, each problem in the batch moves on independently, and every iteration does the same amount of work whichever branch it takes.

## Solver parameters

The parameters live in three update rules that `AugmentedLagrangian` holds as fields. Each rule is a small class that satisfies a protocol, so a different rule can replace any of them.

### Penalty

`AlgencanPenaltyUpdate` implements the penalty rule of ALGENCAN [2]. After each inner solve it measures the infeasibility

$$
V = \max\left( \lVert h \rVert_\infty, \; \left\lVert \min\left(-g, \frac{\mu}{\rho}\right) \right\rVert_\infty \right),
$$

which is small when the constraints hold and inactive constraints have small multipliers. If $V$ has not dropped to at most $\tau$ times its previous value, the penalty grows by a factor $\gamma$.

- `growth_factor` is $\gamma$, by default 10. A larger value reaches a sufficient penalty in fewer outer iterations, but it can overshoot and leave the inner problems worse conditioned than they need to be.
- `decrease_factor` is $\tau$, by default 0.5. It sets how much progress counts as enough. A smaller value demands faster progress, so the penalty grows more often.
- `initial_value` is the penalty for the first inner solve. By default it comes from the formula of Birgin and Martínez, $\rho_0 = 10 \max(1, \lvert f \rvert) / \max(1, \frac{1}{2} \lVert h \rVert^2 + \frac{1}{2} \lVert \max(0, g) \rVert^2)$ at `y0`, clipped to $[10^{-8}, 10^{8}]$. This balances the objective against the initial violation. When `y0` is nearly feasible and the objective is large, the formula can start with a penalty that is higher than needed, and a fixed value like 1 can be faster.
- `maximum_value` caps the penalty, by default at $10^{12}$ in float64 and $10^{6}$ in float32. When the penalty is at its cap and the infeasibility still stops decreasing, the solve ends with `RESULTS.infeasible`. A lower cap detects infeasible problems sooner, and a higher one gives badly scaled problems more room.
- `per_constraint=True` gives every constraint element its own penalty, which grows based on that element alone. This helps when constraints have very different scales, since one hard constraint no longer drives up the penalty on all the others.

### Multipliers

`SafeguardedMultiplierUpdate` applies the first-order update from above and then clips the multipliers used in the next $L_\rho$ to a box, $[-\bar\lambda, \bar\lambda]$ for equality constraints and $[0, \bar\mu]$ for inequality constraints. The convergence theory of Andreani et al. [1] for nonconvex problems depends on this box. When a multiplier estimate is bad, as it can be far from the solution, the box stops it from sending the next inner solve somewhere unreasonable.

- `equality_bound` and `inequality_bound` set $\bar\lambda$ and $\bar\mu$, by default $10^{20}$ in float64 and $10^{6}$ in float32. A bound smaller than the true multiplier turns the method back into a pure penalty method for that constraint, with the slow convergence that comes with it.

The multipliers in `Solution.multipliers` are the estimates before clipping.

### Inner tolerance

`GeometricToleranceSchedule` sets the `rtol` and `atol` of each inner solve. The first inner solve uses `initial`, by default $10^{-2}$, and each following one multiplies it by `factor`, by default 0.1, until it reaches the `rtol` and `atol` of the inner minimiser. Early inner solves don't need to be accurate, because the multipliers are still far off and the next update will move the target anyway.

- A larger `initial` or smaller `factor` saves inner steps early on, but it gives the penalty rule noisier infeasibility measurements to judge progress by.

### Termination

The solve stops when the inner tolerance has reached its final value, and the largest of $\lVert h \rVert_\infty$ and $\lVert \min(-g, \mu) \rVert_\infty$ is below `feasibility_tolerance`. The second term is small when every inequality holds and every inactive constraint has a zero multiplier.

- `feasibility_tolerance` is $10^{-6}$ in float64 and $10^{-3}$ in float32 by default. A minimiser can only locate $y$ to about the square root of the machine epsilon, because $f$ changes quadratically near a minimum. Asking for much less violation than that makes the penalty grow while $y$ stays where it is, and the multiplier estimates degrade with every growth.
- `inner_max_steps` ends an inner solve after that many steps, by default 256. The multipliers are then updated from wherever the inner minimiser got to, which is often good enough to make progress.

## Differentiation

At a solution, the KKT conditions hold:

$$
\nabla f + \nabla h^\top \lambda + \nabla g^\top \mu = 0, \quad h = 0, \quad g_j = 0 \text{ for active } j, \quad \mu_j = 0 \text{ for inactive } j.
$$

If the problem depends on parameters $\theta$, the implicit function theorem gives the derivatives of $y^*$, $\lambda^*$ and $\mu^*$ with respect to $\theta$ from one linear solve with the Jacobian of these conditions. aulax fixes the active set at the solution, so the system keeps the same shape for every active set. It only includes active multipliers in the gradient term, which makes the Jacobian symmetric:

$$
K = \begin{bmatrix} H & \nabla h^\top & \nabla g_{\mathcal{A}}^\top & 0 \\ \nabla h & 0 & 0 & 0 \\ \nabla g_{\mathcal{A}} & 0 & 0 & 0 \\ 0 & 0 & 0 & -I \end{bmatrix},
$$

where $H$ is the Hessian of the Lagrangian and $\mathcal{A}$ is the active set. A constraint with $g_j = 0$ and $\mu_j = 0$ at the solution counts as inactive. The derivative is not unique at such a point, and this picks one of the valid ones.

## References

1. R. Andreani, E. G. Birgin, J. M. Martínez, M. L. Schuverdt. On augmented Lagrangian methods with general lower-level constraints. SIAM J. Optim. 18(4), 2008.
2. E. G. Birgin, J. M. Martínez. Practical Augmented Lagrangian Methods for Constrained Optimization. SIAM, 2014.
3. J. Nocedal, S. J. Wright. Numerical Optimization, 2nd ed. Springer, 2006. Chapter 17.
4. M. J. D. Powell. A method for nonlinear constraints in minimization problems. In R. Fletcher (ed.), Optimization, Academic Press, 1969.
5. R. T. Rockafellar. A dual approach to solving nonlinear programming problems by unconstrained optimization. Math. Program. 5, 1973.
