# Theory

This page explains the method behind aulax, starting from the basics of constrained optimization, and then goes through what each solver parameter does.

## The problem

When doing constrained optimization, we'd like to solve a problem of the form:

$$
\min_y f(y) \quad \text{subject to} \quad h(y) = 0, \quad g(y) \le 0.
$$

Here $f$ is the objective, and $h$ and $g$ return the equality and inequality residuals, one value per constraint. The points that satisfy all constraints form the feasible set, and we are looking for the lowest point of $f$ within that set. Usually the lowest point of $f$ overall lies outside it, which is why an unconstrained minimiser can't be used directly.

Let's first look at equality constraints only. At a solution $y^*$, we can't lower $f$ by moving along the constraints, so the gradient of $f$ has to point straight across them. In other words, it is a combination of the constraint gradients:

$$
\nabla f(y^*) + \nabla h(y^*)^\top \lambda^* = 0.
$$

The vector $\lambda^*$ holds the Lagrange multipliers, one per constraint. A multiplier tells us how much the optimal objective would change if we loosened its constraint a little, which is why it is also called a shadow price.

There are two classic ways to turn the constrained problem into an unconstrained one:

1. **Penalty method.** Add a term to the objective that penalizes constraint violations:
   $$
   f(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
   $$
2. **Lagrangian method.** Add the constraints to the objective, weighted by the multipliers:
   $$
   L(y, \lambda) = f(y) + \lambda^\top h(y).
   $$
   The condition above is then $\nabla_y L = 0$, together with $h = 0$.

Both have a problem. The minimizer of the penalty function never quite satisfies the constraints. Setting its gradient to zero gives $\nabla f + \rho \nabla h^\top h = 0$, and comparing this with the condition above shows that $\rho \, h \approx \lambda^*$. The violation is therefore about $\lambda^* / \rho$, and it only goes to zero as $\rho$ goes to infinity. A large $\rho$ makes the function very steep across the constraints and flat along them, which is hard for any minimiser. The Lagrangian, on the other hand, has the right stationary point, but the solution is generally a saddle point of $L$ and not a minimum, so we can't find it by minimizing.

## The augmented Lagrangian method

The two approaches can be combined into an "augmented Lagrangian" function:

$$
L_\rho(y; \lambda) = f(y) + \lambda^\top h(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

The multipliers let the solution satisfy the constraints exactly, while the penalty term pulls the minimizer towards them and makes the solution a minimum rather than a saddle point. At a minimizer of $L_\rho$, the gradient is

$$
\nabla f(y) + \nabla h(y)^\top \big( \lambda + \rho \, h(y) \big) = 0,
$$

which looks like the optimality condition with $\lambda + \rho \, h(y)$ in place of the multipliers. That suggests the following algorithm:

1. Minimize $L_\rho(y; \lambda)$ over $y$, starting from the current $y$.
2. Update the multipliers with $\lambda \leftarrow \lambda + \rho \, h(y)$.
3. Repeat until the constraints hold.

We call step 1 the inner solve, and one pass through all three steps an outer iteration. The method is useful because the hard part is an unconstrained minimization, which any unconstrained minimiser can do. Once $\lambda$ is close to $\lambda^*$, the minimizer of $L_\rho$ satisfies the constraints for a finite $\rho$, so the penalty only has to be large enough, not infinite.

## The augmented Lagrangian as a shifted penalty

The algorithm is simple, but it can be hard to say what the multipliers do in an intuitive way. Completing the square helps:

$$
L_\rho(y; \lambda) = f(y) + \frac{\rho}{2} \left\lVert h(y) + \frac{\lambda}{\rho} \right\rVert^2 - \frac{\lVert \lambda \rVert^2}{2\rho}.
$$

The last term does not depend on $y$, so minimizing $L_\rho$ is the same as minimizing a penalty function whose target is shifted from $h = 0$ to $h = -\lambda / \rho$. This makes sense if we remember that the penalty method misses its target by about $\lambda^* / \rho$. By aiming past the constraint by exactly that amount, the minimizer lands on $h = 0$. The multiplier update is what finds the right shift. The article [Augmented Lagrangian methods as modified penalty methods](https://nardilam.nl/articles/alm-penalty/) works out this view in more detail.

## Inequality constraints

An inequality constraint $g_j(y) \le 0$ is either active at the solution, when it holds with equality, or inactive. An active constraint behaves like an equality constraint with a multiplier $\mu_j \ge 0$. The sign is fixed because an inequality can only push the solution back into the feasible set, never pull it out. An inactive constraint has $\mu_j = 0$. Together, this gives the Karush-Kuhn-Tucker (KKT) conditions for a solution:

$$
\nabla f + \nabla h^\top \lambda + \nabla g^\top \mu = 0, \quad h = 0, \quad g \le 0, \quad \mu \ge 0, \quad \mu_j g_j = 0.
$$

For inequalities, the Powell-Hestenes-Rockafellar form of the augmented Lagrangian adds, up to a constant, a one-sided shifted penalty:

$$
\frac{\rho}{2} \left\lVert \max\left(0, g(y) + \frac{\mu}{\rho}\right) \right\rVert^2,
$$

with the update $\mu \leftarrow \max(0, \mu + \rho \, g(y))$. This term is zero while $g_j \le -\mu_j / \rho$, and it grows quadratically beyond that point. So a constraint with a positive multiplier gets penalized a little before its boundary, which holds the minimizer right on it, and a constraint with $\mu_j = 0$ is left alone until it is violated. The method sorts out which constraints are active by itself.

There is a numerical cost to the one-sided penalty. Its second derivative jumps from zero to $\rho$ at $g_j = -\mu_j / \rho$. Quasi-Newton minimisers like BFGS build their Hessian approximation from gradient differences, and they lose accuracy when many constraints cross that point during an inner solve. That is why problems with many bound constraints need more inner steps.

## The practical method

The algorithm above leaves open how large $\rho$ should be and how accurately to do each inner solve. aulax follows ALGENCAN [2], which decides both as it goes. The penalty only grows when the constraint violation stops decreasing fast enough, since good multipliers make a large penalty unnecessary. The early inner solves are loose, since the multipliers are still far off and an accurate minimizer would be wasted. The multipliers that go into the next $L_\rho$ are also clipped to a large box, so that one bad estimate far from the solution can't send the next inner solve too far away. The [solver parameters](#solver-parameters) section describes each of these.

## The solver loop

aulax minimizes each $L_\rho$ with an optimistix minimiser, which it calls one step at a time inside a single `jax.lax.while_loop`. When the inner minimiser reports that it has converged, the next loop iteration updates the multipliers and the penalty, and restarts the inner minimiser at the same point. The update comes one iteration late, because an optimistix minimiser returns the evaluation of the point it started its step from. To get $h$ and $g$ at the inner solution, the loop takes one more step from there, uses the evaluation that comes back, and discards the step itself.

A single loop matters under `jax.vmap`. With an outer loop around complete inner solves, every problem in a batch would wait for the slowest inner solve before any of them could update its multipliers. In a single loop, each problem moves on by itself.

## Solver parameters

The parameters live in three update rules that `AugmentedLagrangian` holds as fields. Each rule is a small class that satisfies a protocol, so a different rule can replace any of them.

### Penalty

`AlgencanPenaltyUpdate` implements the penalty rule of ALGENCAN [2]. After each inner solve it measures the infeasibility

$$
V = \max\left( \lVert h \rVert_\infty, \; \left\lVert \min\left(-g, \frac{\mu}{\rho}\right) \right\rVert_\infty \right),
$$

which is small when the constraints hold and inactive constraints have small multipliers. If $V$ has not dropped to at most $\tau$ times its previous value, the penalty grows by a factor $\gamma$.

- `growth_factor` is $\gamma$, by default 10. A larger value reaches a sufficient penalty in fewer outer iterations, but it can overshoot and make the inner problems harder than they need to be.
- `decrease_factor` is $\tau$, by default 0.5. It sets how much progress counts as enough. A smaller value asks for faster progress, so the penalty grows more often.
- `initial_value` is the penalty for the first inner solve. By default it comes from the formula of Birgin and Martínez, $\rho_0 = 10 \max(1, \lvert f \rvert) / \max(1, \frac{1}{2} \lVert h \rVert^2 + \frac{1}{2} \lVert \max(0, g) \rVert^2)$ at `y0`, clipped to $[10^{-8}, 10^{8}]$. This balances the objective against the initial violation. When `y0` is nearly feasible and the objective is large, the formula can start higher than needed, and a fixed value like 1 can be faster.
- `maximum_value` caps the penalty, by default at $10^{12}$ in float64 and $10^{6}$ in float32. When the penalty is at its cap and the infeasibility stops decreasing, the solve ends with `RESULTS.infeasible`. A lower cap detects infeasible problems sooner, and a higher one gives badly scaled problems more room.
- `per_constraint=True` gives every constraint element its own penalty, which grows based on that element alone. This helps when constraints have very different scales, since one hard constraint no longer drives up the penalty on all the others.

### Multipliers

`SafeguardedMultiplierUpdate` applies the multiplier updates from above, and then clips the multipliers used in the next $L_\rho$ to $[-\bar\lambda, \bar\lambda]$ for equality constraints and $[0, \bar\mu]$ for inequality constraints. The convergence theory of Andreani et al. [1] for nonconvex problems depends on this box.

- `equality_bound` and `inequality_bound` set $\bar\lambda$ and $\bar\mu$, by default $10^{20}$ in float64 and $10^{6}$ in float32. A bound smaller than the true multiplier turns the method back into a plain penalty method for that constraint, with its slow convergence.

The multipliers in `Solution.multipliers` are the estimates before clipping.

### Inner tolerance

`GeometricToleranceSchedule` sets the `rtol` and `atol` of each inner solve. The first inner solve uses `initial`, by default $10^{-2}$, and each following one multiplies it by `factor`, by default 0.1, until it reaches the `rtol` and `atol` of the inner minimiser. A larger `initial` or smaller `factor` saves inner steps early on, but it gives the penalty rule noisier measurements to judge progress by.

### Termination

The solve stops when the inner tolerance has reached its final value, and the largest of $\lVert h \rVert_\infty$ and $\lVert \min(-g, \mu) \rVert_\infty$ is below `feasibility_tolerance`. The second term is small when every inequality holds and every inactive constraint has a zero multiplier.

- `feasibility_tolerance` is $10^{-6}$ in float64 and $10^{-3}$ in float32 by default. A minimiser can only locate $y$ to about the square root of the machine epsilon, because $f$ changes quadratically near a minimum. Asking for much less violation than that makes the penalty grow while $y$ stays where it is, and the multiplier estimates get worse with every increase.
- `inner_max_steps` ends an inner solve after that many steps, by default 256. The multipliers are then updated from wherever the inner minimiser got to, which is often good enough to make progress.

## Differentiation

When a problem depends on parameters $\theta$ through `args`, we often want the derivative of the solution with respect to them. We could differentiate through every step of the solver, but that stores every step, and it differentiates the path the solver took rather than the solution. Instead, we can use the fact that the KKT conditions hold at the solution for every $\theta$. Differentiating them with respect to $\theta$ gives one linear system for the derivatives of $y^*$, $\lambda^*$ and $\mu^*$. This is the implicit function theorem, and its cost does not depend on how many steps the solve took.

aulax fixes the active set $\mathcal{A}$ at the solution, and only includes the multipliers of active constraints in the gradient term. The matrix of the linear system is then symmetric:

$$
K = \begin{bmatrix} H & \nabla h^\top & \nabla g_{\mathcal{A}}^\top & 0 \\ \nabla h & 0 & 0 & 0 \\ \nabla g_{\mathcal{A}} & 0 & 0 & 0 \\ 0 & 0 & 0 & -I \end{bmatrix},
$$

where $H$ is the Hessian of the Lagrangian, and the last block row sets the derivatives of the inactive multipliers to zero. A constraint with $g_j = 0$ and $\mu_j = 0$ at the solution counts as inactive. The derivative is not unique at such a point, and this picks one of the valid ones.

## References

1. R. Andreani, E. G. Birgin, J. M. Martínez, M. L. Schuverdt. On augmented Lagrangian methods with general lower-level constraints. SIAM J. Optim. 18(4), 2008.
2. E. G. Birgin, J. M. Martínez. Practical Augmented Lagrangian Methods for Constrained Optimization. SIAM, 2014.
3. J. Nocedal, S. J. Wright. Numerical Optimization, 2nd ed. Springer, 2006. Chapter 17.
4. M. J. D. Powell. A method for nonlinear constraints in minimization problems. In R. Fletcher (ed.), Optimization, Academic Press, 1969.
5. R. T. Rockafellar. A dual approach to solving nonlinear programming problems by unconstrained optimization. Math. Program. 5, 1973.
