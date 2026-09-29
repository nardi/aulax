# Theory

This page explains how aulax solves constrained problems. It assumes you know what a gradient is and have seen an unconstrained minimiser like gradient descent or BFGS at work, but not that you have met constrained optimization before. We build the method up from small examples, and the last sections connect it to the parameters you can set in aulax.

## Constrained problems

An unconstrained minimiser looks for the lowest point of a function $f(y)$ anywhere. In many problems, though, only some points are allowed. A trajectory has to obey the laws of motion, a design has to stay within its material limits, and a probability vector has to sum to one. We write such a problem as

$$
\min_y f(y) \quad \text{subject to} \quad h(y) = 0, \quad g(y) \le 0.
$$

Here $y$ collects all variables, $f$ is the objective, $h$ returns the equality residuals, and $g$ returns the inequality residuals. Both $h$ and $g$ can return many values, one per constraint. The points that satisfy all constraints form the feasible set, and we want the lowest point of $f$ within it.

The [basic usage](guide/basic-usage.md) page has an example that is easy to picture. We look for the point on the unit circle closest to $p = (2, 1)$, with its second coordinate at most $0.3$:

$$
f(y) = \lVert y - p \rVert^2, \quad h(y) = y_1^2 + y_2^2 - 1, \quad g(y) = y_2 - 0.3.
$$

The unconstrained minimum of $f$ is $p$ itself, which lies far off the circle. So the answer is decided by the constraints, not by $f$ alone, and a minimiser that ignores them will not find it.

## Conditions at a solution

Before we look at methods, it helps to know what a solution looks like, because the methods are built around it.

Think of the circle without the inequality first. At the closest point on the circle, moving along the circle in either direction makes $f$ larger. That means the gradient $\nabla f$ has no component along the circle. It must point straight out of the circle or straight into it, which is the direction of the circle's own normal $\nabla h$. So at the solution, $\nabla f$ is a multiple of $\nabla h$, and we can write that as

$$
\nabla f(y) + \lambda \nabla h(y) = 0
$$

for some number $\lambda$. This number is the Lagrange multiplier of the constraint. With many equality constraints there is one multiplier per constraint, and the condition becomes $\nabla f + \nabla h^\top \lambda = 0$. The function

$$
L(y, \lambda) = f(y) + \lambda^\top h(y)
$$

is the Lagrangian. Its gradient in $y$ is exactly the expression above, so a solution is a point where $h(y) = 0$ and $\nabla_y L = 0$.

The multiplier has a meaning of its own. It measures how much the optimal objective would change if we moved the constraint a little. Changing the constraint from $h(y) = 0$ to $h(y) = \varepsilon$ changes the optimal $f$ by about $-\lambda \varepsilon$. Economists call this a shadow price. In the circle example, it tells us how much closer to $p$ we could get with a slightly larger circle.

Inequality constraints add one twist. At the solution, an inequality either holds with equality, and then it is active and acts like an equality constraint, or it holds strictly, and then it is inactive and plays no role at all. An inactive constraint gets a zero multiplier. An active one gets a multiplier $\mu_j \ge 0$, and the sign matters. An inequality can only push the solution towards its feasible side, never pull it outward. Together these are the Karush-Kuhn-Tucker (KKT) conditions:

$$
\nabla f + \nabla h^\top \lambda + \nabla g^\top \mu = 0, \quad h = 0, \quad g \le 0, \quad \mu \ge 0, \quad \mu_j g_j = 0 \text{ for every } j.
$$

The last condition, complementarity, says that for each inequality either the multiplier or the residual is zero. For the circle example, aulax finds $y^* \approx (0.954, 0.3)$ with $\lambda \approx 1.097$ and $\mu \approx 0.742$. You can check that $\nabla f = 2(y^* - p)$ is cancelled by $\lambda \nabla h = 2 \lambda y^*$ and $\mu \nabla g = (0, \mu)$.

Every method in this field looks for a point and multipliers that satisfy these conditions. They differ in how they get there.

## Penalty method

The simplest idea turns the constrained problem into an unconstrained one by charging for violations. We add a penalty to the objective:

$$
\min_y f(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

Now any unconstrained minimiser applies. The catch shows up in a one-dimensional example. Take $f(x) = x^2$ with the constraint $h(x) = x - 1 = 0$. The solution is $x^* = 1$, and from $2x + \lambda = 0$ we get $\lambda^* = -2$. The penalty function $x^2 + \frac{\rho}{2}(x - 1)^2$ has its minimum at

$$
x_\rho = \frac{\rho}{2 + \rho}.
$$

For $\rho = 10$ that is $0.833$, for $\rho = 100$ it is $0.980$, and for $\rho = 1000$ it is $0.998$. The minimizer only reaches the constraint in the limit. The objective keeps pulling $x$ towards zero, and the penalty only pushes back once there is a violation to push against.

This holds in general. At the minimizer of the penalty function the gradient is zero, so $\nabla f + \rho \nabla h^\top h = 0$. Compare that with the KKT condition $\nabla f + \nabla h^\top \lambda^* = 0$, and you see that $\rho \, h(y_\rho)$ plays the role of $\lambda^*$. The violation is therefore about $\lambda^* / \rho$, and it only becomes small when $\rho$ becomes large.

A large $\rho$ has a price, though. The second derivative of the penalty function above is $2 + \rho$, while that of $f$ is $2$. In more dimensions, the curvature across the constraint grows with $\rho$ while the curvature along it does not. That mismatch makes every unconstrained minimiser slower or less accurate.

## Augmented Lagrangian

The penalty method misses the constraint by $\lambda^* / \rho$ because it has no multiplier. The Lagrangian has the multiplier, but minimizing it over $y$ does not work either, since the solution is generally a saddle point of $L$ and not a minimum. The augmented Lagrangian combines the two:

$$
L_\rho(y; \lambda) = f(y) + \lambda^\top h(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

We can read it either as the Lagrangian with an extra penalty on violations, or as the penalty function turned into a Lagrangian. Its gradient is

$$
\nabla_y L_\rho = \nabla f(y) + \nabla h(y)^\top \big( \lambda + \rho \, h(y) \big).
$$

At a minimizer of $L_\rho$ this is zero, so the vector $\lambda + \rho \, h(y)$ plays the role of the multipliers in the KKT condition. That suggests the update rule

$$
\lambda \leftarrow \lambda + \rho \, h(y),
$$

and the augmented Lagrangian method alternates between two steps. It minimizes $L_\rho$ over $y$ with the current $\lambda$, and then it updates $\lambda$ from the minimizer. We call the minimization the inner solve, and one round of both steps an outer iteration.

In the one-dimensional example, the minimizer of $x^2 + \lambda(x - 1) + \frac{\rho}{2}(x - 1)^2$ is $x = (\rho - \lambda) / (2 + \rho)$. Starting from $\lambda = 0$ with a fixed $\rho = 10$, the first outer iterations give:

| Outer iteration | $x$ | $\lambda$ after the update |
|---|---|---|
| 1 | 0.833 | -1.667 |
| 2 | 0.972 | -1.944 |
| 3 | 0.995 | -1.991 |
| 4 | 0.999 | -1.998 |

After three outer iterations, $x$ is as close to the constraint as the penalty method gets with $\rho = 400$, while $\rho$ stayed at 10. Here the error in $\lambda$ shrinks by a factor $2 / (2 + \rho)$ in every outer iteration. In general, the method converges once $\rho$ is large enough to make $L_\rho$ convex near the solution, and a larger $\rho$ makes each outer iteration gain more. The penalty no longer has to grow without bound. It only has to be large enough.

The same example in aulax finds the multiplier we derived by hand:

<!-- name: test_theory -->
```python
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def one_dimensional(y, args, state):
    return alx.Evaluation(objective=y[0] ** 2, equality=y[0] - 1.0)


solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
solution = alx.minimise(one_dimensional, solver, jnp.zeros(1))
assert jnp.allclose(solution.value, 1.0, atol=1e-5)
assert jnp.allclose(solution.multipliers.equality, -2.0, atol=1e-4)
```

## Shifted-penalty view

There is a second way to see why the multiplier helps. Completing the square gives

$$
L_\rho(y; \lambda) = f(y) + \frac{\rho}{2} \left\lVert h(y) + \frac{\lambda}{\rho} \right\rVert^2 - \frac{\lVert \lambda \rVert^2}{2\rho}.
$$

The last term does not depend on $y$, so minimizing $L_\rho$ is the same as minimizing a penalty function whose target is shifted from $h = 0$ to $h = -\lambda / \rho$. We saw that the plain penalty method lands short of its target by about $\lambda^* / \rho$. If we aim past the constraint by exactly that amount, the minimizer lands on $h = 0$, and the multiplier update is what finds the right amount. In the one-dimensional example with $\rho = 10$, the final target is $x - 1 = -\lambda^* / \rho = 0.2$. The penalty pulls towards $x = 1.2$, the objective pulls towards $x = 0$, and the two balance at exactly $x = 1$. The article [Augmented Lagrangian methods as modified penalty methods](https://nardilam.nl/articles/alm-penalty/) works this view out in more detail.

## Inequality constraints

An inequality constraint $g_j(y) \le 0$ should only be penalized when it is violated, or close to it. With multipliers $\mu \ge 0$, the Powell-Hestenes-Rockafellar form of the augmented Lagrangian is

$$
L_\rho(y; \lambda, \mu) = f + \lambda^\top h + \frac{\rho}{2} \lVert h \rVert^2 + \frac{1}{2\rho} \left( \lVert \max(0, \mu + \rho g) \rVert^2 - \lVert \mu \rVert^2 \right),
$$

and the multiplier update becomes $\mu \leftarrow \max(0, \mu + \rho g)$. The $\max$ keeps the multipliers non-negative, as the KKT conditions require.

The shifted-penalty view makes this easier to read. Up to a constant, the inequality term is $\frac{\rho}{2} \lVert \max(0, g + \mu / \rho) \rVert^2$. That is a one-sided penalty, which is zero while $g_j \le -\mu_j / \rho$ and grows quadratically beyond that point. A constraint with a positive multiplier starts being penalized a little before its boundary, so the penalty holds the minimizer right on the boundary. A constraint that is comfortably satisfied ends up with $\mu_j = 0$, and then it is only penalized once violated, which does not happen at the solution. So the method sorts the constraints into active and inactive by itself, without being told which is which.

The one-sided penalty has a price. Its second derivative jumps from zero to $\rho$ at $g_j = -\mu_j / \rho$. Quasi-Newton minimisers like BFGS build their Hessian approximation from gradient differences, and they lose accuracy when many constraints cross that point during an inner solve. Problems with many bound constraints need more inner steps for this reason.

## Practical method

The method as described so far leaves open how large $\rho$ should be and how accurately each inner solve should be done. Practical implementations decide both as they go.

- The penalty starts moderate and grows only when the constraint violation stops decreasing fast enough. A good multiplier estimate makes a large penalty unnecessary, so the penalty only grows when the multipliers alone are not making progress.
- Early inner solves use a loose tolerance. The multipliers are still far off at that point, and the next update will move the target anyway, so an accurate minimizer would be wasted work.
- The multipliers that go into the next $L_\rho$ are clipped to a large box. Far from the solution, a multiplier estimate can be poor, and the box stops one bad estimate from sending the next inner solve far away.

aulax follows ALGENCAN [2] in all three, and the [solver parameters](#solver-parameters) section below describes each of them.

## Solver loop

This section is about how aulax runs the method, rather than the method itself, so you can skip it on a first read.

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

Often a problem depends on parameters $\theta$, passed in through `args`, and we want to know how the solution changes with them. We might fit $\theta$ to data by gradient descent, where each evaluation of the loss involves a constrained solve, or want the sensitivity of a design to its specifications.

One way to get that derivative is to differentiate through every step of the solver. That works, but it has to store every step, and it differentiates how the solver got to the solution rather than the solution itself. There is a shorter way. The solution is defined by the KKT conditions, and those hold for every $\theta$. If we write them as $F(y^*(\theta), \lambda^*(\theta), \theta) = 0$ and differentiate both sides, we get a linear equation for the derivatives of $y^*$ and $\lambda^*$. This is the implicit function theorem.

The one-dimensional example shows how this works. Change the constraint to $x = \theta$. The KKT conditions are $2x + \lambda = 0$ and $x - \theta = 0$. Differentiating both with respect to $\theta$ gives $2 x' + \lambda' = 0$ and $x' = 1$, so $x' = 1$ and $\lambda' = -2$. That matches the solution $x^* = \theta$, $\lambda^* = -2\theta$, and we never had to look at the iterations.

For a general problem, the KKT conditions at a solution are

$$
\nabla f + \nabla h^\top \lambda + \nabla g^\top \mu = 0, \quad h = 0, \quad g_j = 0 \text{ for active } j, \quad \mu_j = 0 \text{ for inactive } j.
$$

Differentiating them with respect to $\theta$ gives one linear system for the derivatives of $y^*$, $\lambda^*$ and $\mu^*$, whose matrix is the Jacobian of these conditions. Its cost does not depend on how many steps the solve took. aulax fixes the active set at the solution, so the system keeps the same shape whichever constraints are active. It only includes active multipliers in the gradient term, which makes the matrix symmetric:

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
