class Value:
    def __init__(self, n, children=(), op=''):
        self.data = n
        self.grad = 0
        self.op = op
        self.prev = set(children)
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data + other.data, (self, other), '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def backward():
            self.grad += other * (self.data**(other-1)) * out.grad

        out._backward = backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), 'tanh')

        def backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = backward
        return out

    def __neg__(self):
        return Value(-self.data)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return other * self

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

        
class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
        out = act.tanh()
        return out

    def __repr__(self):
        return f'Neuron({len(self.w)})'

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def __repr__(self):
        return f"Layer of [{',\n\t'.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def parameters(self):
        return [p for layer in self.layers  for p in layer.parameters()]

    def __repr__(self):
        return f'MLP of [\n{', \n\t\n'.join(str(layer) for layer in self.layers)}]'

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
