# aulax

aulax is a JAX library for constrained nonlinear optimization with the augmented Lagrangian method. It minimizes an objective subject to equality and inequality constraints, and it uses the unconstrained minimisers from [optimistix](https://github.com/patrick-kidger/optimistix) for the inner solves.

Install with pip:

```bash
pip install aulax
```

and [read the documentation here](https://nardi.github.io/aulax). The conventional import style is `import aulax as alx`.

aulax is in early development, and nothing in it is stable yet.

## License

aulax is licensed under the [Hippocratic License 3.0](LICENSE.md).
