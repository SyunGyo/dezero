import numpy as np
import weakref
import contextlib
import os
import subprocess
import dezero

@contextlib.contextmanager
def using_config(name,value):
    old_value = getattr(Config,name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

class Config:
    enable_backprop = True

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)


        add_func(self.creator)

        while funcs:
            func = funcs.pop()

            gys = [ output().grad for output  in func.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = func.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x , gx in zip(func.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:
                    for y in func.outputs:
                        y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        return dezero.functions.reshape(self,shape)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
class Parameter(Variable):
    pass

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name,value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outpus = [weakref.ref(y) for y in outputs] 

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()



class Function:
    def __call__(self, *inputs):

        inputs = [as_variable(as_array(x)) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy 
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        return gx0, gx1
    
def add(x0, x1):
    f = Add()
    return f(x0, x1)
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = np.exp(x) * gy
        return gx
    
def exp(x):
    f = Exp()
    return f(x)

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0
    
def mul(x0,x1):
    f = Mul()
    return f(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    f = Neg()
    return f(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # x1, x0を入れ替え



class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # x1, x0を入れ替え


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
def pow(x, c):
    return Pow(c)(x)



def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    
    return Variable(obj)

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.data) + ' ' + str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))

    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))


    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set([])
    def add_func(f):
        if not f in seen_set:
            funcs.append(f)
            seen_set.add(f)

    txt += _dot_var(output, verbose=verbose)
    add_func(output.creator)
    while funcs:
        f = funcs.pop()
        inputs = f.inputs

        txt += _dot_func(f)

        for input in inputs:
            txt += _dot_var(input, verbose=verbose)
            if input.creator is not None:
                add_func(input.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir=os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)
            



def setup_variable():
    Variable.__add__  = add
    Variable.__radd__ = add
    Variable.__mul__  = mul
    Variable.__rmul__ = mul
    Variable.__neg__  = neg
    Variable.__sub__  = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__  = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__  = pow