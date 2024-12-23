import numpy as np

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)


        funcs = [self.creator]

        while funcs:
            func = funcs.pop()

            x, y = func.input, func.output

            x.grad = func.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)

        self.input = input
        self.output = output

        return output
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data 
        gx = np.exp(x) * gy
        return gx
    
def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    
    return x
