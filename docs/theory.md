# Theory

This page explains the method behind aulax, starting from the basics of constrained optimization, and then describes each solver parameter.

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

The vector $\lambda^*$ contains the Lagrange multipliers, one per constraint. A multiplier is the rate at which the optimal objective changes when its constraint is loosened a little, which is why it is also called a shadow price.

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

The penalty method has the drawback that the minimizer of the penalty function never quite satisfies the constraints. Setting its gradient to zero gives $\nabla f + \rho \nabla h^\top h = 0$, and comparing this with the condition above shows that $\rho \, h \approx \lambda^*$. The violation is therefore about $\lambda^* / \rho$, and it only goes to zero as $\rho$ goes to infinity. A large $\rho$ makes the function very steep across the constraints and flat along them, which is hard for any minimiser. The Lagrangian, on the other hand, has the right stationary point, but the solution is generally a saddle point of $L$ and not a minimum, so we can't find it by minimizing.

## The augmented Lagrangian method

The two approaches can be combined into an "augmented Lagrangian" function:

$$
L_\rho(y; \lambda) = f(y) + \lambda^\top h(y) + \frac{\rho}{2} \lVert h(y) \rVert^2.
$$

With the multipliers, the minimizer can satisfy the constraints exactly for a finite penalty, and the penalty term makes the solution a minimum rather than a saddle point. At a minimizer of $L_\rho$, the gradient is

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

The last term does not depend on $y$, so minimizing $L_\rho$ is the same as minimizing a penalty function whose target is shifted from $h = 0$ to $h = -\lambda / \rho$. This makes sense if we remember that the penalty method misses its target by about $\lambda^* / \rho$. When the target is shifted past the constraint by exactly that amount, the minimizer satisfies $h = 0$. The multiplier updates converge to this shift. The article [Augmented Lagrangian methods as modified penalty methods](https://nardilam.nl/articles/alm-penalty/) derives this view in more detail.

## Inequality constraints

An inequality constraint $g_j(y) \le 0$ is either active at the solution, when it holds with equality, or inactive. An active constraint behaves like an equality constraint with a multiplier $\mu_j \ge 0$. The sign is fixed because an inequality can only push the solution back into the feasible set, never pull it out. An inactive constraint has $\mu_j = 0$. Together, this gives the Karush-Kuhn-Tucker (KKT) conditions for a solution:

$$
\nabla f + \nabla h^\top \lambda + \nabla g^\top \mu = 0, \quad h = 0, \quad g \le 0, \quad \mu \ge 0, \quad \mu_j g_j = 0.
$$

For inequalities, the Powell-Hestenes-Rockafellar form of the augmented Lagrangian adds, up to a constant, a one-sided shifted penalty:

$$
\frac{\rho}{2} \left\lVert \max\left(0, g(y) + \frac{\mu}{\rho}\right) \right\rVert^2,
$$

with the update $\mu \leftarrow \max(0, \mu + \rho \, g(y))$. This term is zero while $g_j \le -\mu_j / \rho$, and it grows quadratically beyond that point. So a constraint with a positive multiplier is penalized from a point slightly before its boundary, which places the minimizer exactly on the boundary. The penalty term of a constraint with $\mu_j = 0$ is zero until the constraint is violated. The active set is therefore determined by the multiplier updates, without a separate step that selects it.

The one-sided penalty has a numerical drawback, because its second derivative jumps from zero to $\rho$ at $g_j = -\mu_j / \rho$. Quasi-Newton minimisers like BFGS build their Hessian approximation from gradient differences, and they lose accuracy when many constraints cross that point during an inner solve. That is why problems with many bound constraints need more inner steps.

## The practical method

The algorithm above leaves open how large $\rho$ should be and how accurately to do each inner solve. aulax uses the rules of ALGENCAN [2], which set both during the solve. The penalty only grows when the constraint violation stops decreasing fast enough, since good multipliers make a large penalty unnecessary. The early inner solves are loose, since the multipliers are still far off and an accurate minimizer would be wasted. The multipliers that go into the next $L_\rho$ are also clipped to a large box, so that one bad estimate far from the solution has a bounded effect on the next inner solve. The [solver parameters](#solver-parameters) section describes each of these.

## The solver loop

aulax minimizes each $L_\rho$ with an inner minimiser, which is `Newton` by default or any optimistix minimiser. It calls the minimiser one step at a time inside a single `jax.lax.while_loop`. When the inner minimiser reports that it has converged, the next loop iteration updates the multipliers and the penalty, and restarts the inner minimiser at the same point. `Newton` and the optimistix minimisers `BFGS`, `DFP`, `LBFGS`, `GradientDescent` and `NonlinearCG` store the evaluation at their accepted point in their state, so the multipliers are updated in the iteration where the inner solve ends. Other minimisers, such as `OptaxMinimiser`, only return the evaluation of the point that their step started from. To get $h$ and $g$ at the inner solution, the loop then takes one more step from there, uses the evaluation that comes back, and discards the step itself, so their update comes one iteration late.

The single loop matters under `jax.vmap`, because with an outer loop around complete inner solves, every problem in a batch would wait for the slowest inner solve before any of them could update its multipliers. In a single loop, each problem updates its multipliers as soon as its own inner solve ends.

## Solver parameters

The parameters are fields of three update rules, which are in turn fields of `AugmentedLagrangian`. Each rule is a small class that satisfies a protocol, so any of them can be replaced by another class that satisfies the same protocol.

### Penalty

`AlgencanPenaltyUpdate` implements the penalty rule of ALGENCAN [2]. After each inner solve it computes the infeasibility

$$
V = \max\left( \lVert h \rVert_\infty, \; \left\lVert \min\left(-g, \frac{\mu}{\rho}\right) \right\rVert_\infty \right),
$$

which is small when the constraints hold and inactive constraints have small multipliers. If $V$ has not dropped to at most $\tau$ times its previous value, the penalty grows by a factor $\gamma$.

- `growth_factor` is $\gamma$, by default 10. A larger value reaches a sufficient penalty in fewer outer iterations, but it can make the penalty larger than needed, which makes the inner problems harder.
- `decrease_factor` is $\tau$, by default 0.5. It is the required reduction of $V$ from one outer iteration to the next. A smaller value asks for faster progress, so the penalty grows more often.
- `initial_value` is the penalty for the first inner solve. By default it is computed with the formula of Birgin and Martínez, $\rho_0 = 10 \max(1, \lvert f \rvert) / \max(1, \frac{1}{2} \lVert h \rVert^2 + \frac{1}{2} \lVert \max(0, g) \rVert^2)$ at `y0`, clipped to $[10^{-8}, 10^{8}]$. This balances the objective against the initial violation. When `y0` is nearly feasible and the objective is large, the formula can give a higher value than needed, and a fixed value like 1 can be faster.
- `maximum_value` caps the penalty, by default at $10^{12}$ in float64 and $10^{6}$ in float32. When the penalty is at its cap and the infeasibility stops decreasing, the solve ends with `RESULTS.infeasible`. With a lower cap, infeasible problems are detected sooner, and with a higher one, badly scaled problems can use larger penalties.
- `per_constraint=True` gives every constraint element its own penalty, which grows based on that element alone. This helps when constraints have very different scales, since one hard constraint no longer raises the penalty on all the others.
- `penalty_freeze_factor` stops the penalty from growing once the infeasibility is within this multiple of `feasibility_tolerance`. The multipliers keep updating while the penalty stays fixed, and they converge without it. The default is 10 for `Newton` and 0.1 for other minimisers. On the Hock-Schittkowski problems, the factor of 10 removed the one failure that the tighter feasibility tolerance caused.

### Multipliers

`SafeguardedMultiplierUpdate` applies the multiplier updates from above, and then clips the multipliers used in the next $L_\rho$ to $[-\bar\lambda, \bar\lambda]$ for equality constraints and $[0, \bar\mu]$ for inequality constraints. The convergence theory of Andreani et al. [1] for nonconvex problems depends on this box.

- `equality_bound` and `inequality_bound` set $\bar\lambda$ and $\bar\mu$, by default $10^{20}$ in float64 and $10^{6}$ in float32. A bound smaller than the true multiplier makes the method a plain penalty method for that constraint, with its slow convergence.

The multipliers in `Solution.multipliers` are the estimates before clipping.

### Inner tolerance

`GeometricToleranceSchedule` sets the `rtol` and `atol` of each inner solve. The first inner solve uses `initial`, by default $10^{-2}$, and each following one multiplies it by `factor`, by default 0.1, until it reaches the `rtol` and `atol` of the inner minimiser. A larger `initial` or smaller `factor` saves inner steps early on, but the infeasibility $V$ that the penalty rule compares is then less accurate.

### Termination

The solve stops when the inner tolerance has reached its final value, and the largest of $\lVert h \rVert_\infty$ and $\lVert \min(-g, \mu) \rVert_\infty$ is below `feasibility_tolerance`. The second term is small when every inequality holds and every inactive constraint has a zero multiplier.

- `feasibility_tolerance` is $10^{-8}$ in float64 for `Newton`, $10^{-6}$ in float64 for other minimisers, and $10^{-3}$ in float32, by default. The violation at the end of a solve is about the final inner tolerance, so a bound far below that cannot be reached. `Newton` converges quadratically near the solution, so its final tolerance of $10^{-8}$ is at the same level as the bound. On the L-BFGS example of the [advanced usage](guide/advanced-usage.md) guide, the tighter tolerance takes about twice as many steps, which is why the quasi-Newton minimisers keep the looser default.
- `inner_max_steps` ends an inner solve after that many steps, by default 256. The multipliers are then updated from the last point of the inner minimiser, which is often good enough to make progress.

## Comparison with IPOPT and NLopt

IPOPT [6] is an interior point method. It replaces the one-sided penalty with a barrier, and it solves one Newton system for the primal and dual variables at each iteration. aulax uses the augmented Lagrangian instead, which has two disadvantages. Each multiplier update starts a new inner solve. Because of the kink of the one-sided penalty, the trust-region Newton steps also shrink whenever a constraint crosses its shifted boundary [7]. In return, aulax never forms a KKT matrix, so its memory use is proportional to the number of variables and constraints.

The benchmarks compare solvers with performance profiles [8]. The numbers below come from the 49 Hock-Schittkowski problems in `benchmarks/problems`. The default for `Newton` stops at a feasibility of $10^{-8}$. On these problems it has no violation above $10^{-8}$ and one objective error above $10^{-6}$ (relative), and the geometric mean of the evaluations of IPOPT is 0.39 times that of aulax. IPOPT has two violations above $10^{-8}$ and one objective error above $10^{-6}$ on the same problems. Problem `hs2` is left out of these counts, since its known objective differs from the result of every solver.

The 142 S2MPJ problems with bounds, which have at most 100 variables, show the same result for feasibility. aulax leaves no violation above $10^{-8}$, against 3 for IPOPT. aulax solves 115 problems, against 117 for IPOPT, and IPOPT needs 0.84 times the evaluations of aulax.

NLopt [9] provides an augmented Lagrangian method, `AUGLAG`, that runs a local minimiser on each subproblem. The benchmarks run it with L-BFGS and with SLSQP as that minimiser, on the same residuals and tolerances as the other solvers. An error is an exception that NLopt raises during a solve, and it counts as not solved. On the Hock-Schittkowski problems, the L-BFGS variant solves 40, raises an error on 9, and needs 1.80 times the evaluations of aulax. The SLSQP variant solves 37, fails on 8, and needs 3.27 times the evaluations. On the S2MPJ problems with bounds, the SLSQP variant solves 113 against 115 for aulax, at 4.02 times the evaluations, and the L-BFGS variant solves 91 with 30 errors. On the problems they solve, both variants keep the violation below $10^{-8}$.

The IPOPT version here is 3.11.9, from the Ubuntu 24.04 package. The benchmark code pins 3.14.20 for Windows, so the counts on other platforms may shift a little.

Sparse mode has a remaining limitation, because the inertia of the Hessian cannot be obtained from its sparse factorization, so its trust-region steps are less reliable than the dense ones. aulax also still fails on the 1000-variable pendulum and Poisson control problems. The benchmark runs IPOPT only up to 300 variables, so there is no IPOPT count for those problems.

## Differentiation

When a problem depends on parameters $\theta$ through `args`, we often want the derivative of the solution with respect to them. We could differentiate through every step of the solver, but that stores every step, and it differentiates the path the solver took rather than the solution. Instead, we can use the fact that the KKT conditions hold at the solution for every $\theta$. Differentiating them with respect to $\theta$ gives one linear system for the derivatives of $y^*$, $\lambda^*$ and $\mu^*$. This is the implicit function theorem, and its cost does not depend on how many steps the solve took.

aulax fixes the active set $\mathcal{A}$ at the solution, and only includes the multipliers of active constraints in the gradient term. The matrix of the linear system is then symmetric:

$$
K = \begin{bmatrix} H & \nabla h^\top & \nabla g_{\mathcal{A}}^\top & 0 \\ \nabla h & 0 & 0 & 0 \\ \nabla g_{\mathcal{A}} & 0 & 0 & 0 \\ 0 & 0 & 0 & -I \end{bmatrix},
$$

where $H$ is the Hessian of the Lagrangian, and the last block row sets the derivatives of the inactive multipliers to zero. A constraint with $g_j = 0$ and $\mu_j = 0$ at the solution counts as inactive. The derivative is not unique at such a point, and this choice gives one of the valid ones.

## References

1. R. Andreani, E. G. Birgin, J. M. Martínez, M. L. Schuverdt. On augmented Lagrangian methods with general lower-level constraints. SIAM J. Optim. 18(4), 2008.
2. E. G. Birgin, J. M. Martínez. Practical Augmented Lagrangian Methods for Constrained Optimization. SIAM, 2014.
3. J. Nocedal, S. J. Wright. Numerical Optimization, 2nd ed. Springer, 2006. Chapter 17.
4. M. J. D. Powell. A method for nonlinear constraints in minimization problems. In R. Fletcher (ed.), Optimization, Academic Press, 1969.
5. R. T. Rockafellar. A dual approach to solving nonlinear programming problems by unconstrained optimization. Math. Program. 5, 1973.
6. A. Wächter, L. T. Biegler. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Math. Program. 106(1), pp. 25-57, 2006.
7. A. R. Conn, N. I. M. Gould, Ph. L. Toint. Trust-Region Methods. SIAM, 2000.
8. E. D. Dolan, J. J. Moré. Benchmarking optimization software with performance profiles. Math. Program. 91(2), pp. 201-213, 2002.
9. S. G. Johnson. The NLopt nonlinear-optimization package. Available at https://nlopt.readthedocs.io.
