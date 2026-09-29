# aulax

aulax is a JAX library for constrained nonlinear optimization. It solves problems of the form

$$
\min_y f(y) \quad \text{subject to} \quad h(y) = 0, \quad g(y) \le 0
$$

with the augmented Lagrangian method, and it uses the unconstrained minimisers from [optimistix](https://github.com/patrick-kidger/optimistix) for the inner solves.

Install it with pip:

```bash
pip install aulax
```

The conventional import style is

<!-- name: test_import -->
```python
import aulax as alx
```

which matches `import optimistix as optx` and `import lineax as lx`.

aulax is licensed under the [Hippocratic License 3.0](https://github.com/nardi/aulax/blob/main/LICENSE.md).
