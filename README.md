# minigrad

A miniature implementation of PyTorch's autograd engine that performs backpropagation over a dynamically built computation graph, with a small neural network library (`Neuron`, `Layer`, `MLP`) built on top of it.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## How it works

At the core is the `Value` class, a wrapper around a single scalar number that:

- Tracks how it was produced (`children`, `op`) so a computation graph is built automatically as you do math on it.
- Supports `+`, `-`, `*`, `/`, `**`, along with `relu()` and `tanh()` activation functions.
- Stores a local `_backward()` function on every operation, which knows how to propagate gradients backward through that specific op.
- Exposes a top-level `.backward()` method that topologically sorts the graph and calls each node's `_backward()` in reverse order, computing gradients for every value that contributed to the final result.

`Neuron`, `Layer`, and `MLP` are built entirely out of `Value` operations, so calling `.backward()` on a loss computed from an `MLP` automatically backpropagates gradients into every weight and bias in the network.

## Installation

No dependencies beyond the Python standard library (`random`, `math`).

```bash
git clone git@github.com:dishanshakya/minigrad.git
cd minigrad
```

## Usage

### Basic autograd

```python
from minigrad import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + a**2
c.backward()

print(a.grad)  # dc/da
print(b.grad)  # dc/db
```

### Building a neural network

```python
from minigrad import MLP

# 4 inputs -> hidden layers of 5, 5, 5 -> 1 output
n = MLP(4, [5, 5, 5, 1])

x = [2.3, 5.6, 7.8, -1.1]
y_pred = n(x)
print(y_pred)
```

### Training loop

```python
xs = [
    [2.3, 5.6, 7.8, -1.1],
    [3.1, 4.3, -2.2, -0.5],
    [6.0, -0.4, 2.1, -2.2],
    [1.1, -3.6, 9.8, 1.0]
]
ys = [1.0, 2.0, 1.0, 4.0]

n = MLP(4, [5, 5, 5, 1])

for k in range(100):
    ypreds = [n(x) for x in xs]
    loss = sum((ygt - ypred)**2 for ygt, ypred in zip(ys, ypreds))

    n.zero_grad()
    loss.backward()

    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)
```

## API reference

| Class | Description |
|---|---|
| `Value(n, children=(), op='')` | Scalar wrapper that tracks computation history for autograd. |
| `Value.backward()` | Computes gradients for every `Value` in the graph leading to this one. |
| `Neuron(nin)` | Single neuron with `nin` weights, a bias, and a `tanh` activation. |
| `Layer(nin, nout)` | A layer of `nout` neurons, each taking `nin` inputs. |
| `MLP(nin, nouts)` | Multi-layer perceptron; `nouts` is a list of layer sizes, e.g. `[5, 5, 1]`. |
| `.parameters()` | Returns a flat list of all `Value` weights/biases in a `Neuron`, `Layer`, or `MLP`. |
| `.zero_grad()` | Resets `.grad` to 0 for all parameters (call before each `backward()`). |

## Notes

- Every output neuron uses `tanh`, which bounds predictions to `(-1, 1)`. If your targets fall outside that range, normalize them first or modify the last layer to skip the activation.
- Gradients accumulate across calls to `.backward()`, so always call `.zero_grad()` before each training step, just like PyTorch's `optimizer.zero_grad()`.

## License

MIT
