from dezero.core import Function, as_variable
import numpy as np
from dezero import utils

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)

        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)

    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)

        return gx
    
def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum()
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x):
    return Sum()(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x,shape):
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)

        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW
    
def matmul(x, W):
    return MatMul()(x,W)

class Exp(Function):
    def forward(self, x):
        y = np.exp(-x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]
        return gy * y
    
def exp(x):
    return Exp()(x)

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))

        return y
    
    def backward(self, gy):
        y = self.outputs[0]()

        return gy * y * (1 - y)
    
def sigmoid(x):
    return Sigmoid()(x)

def linear(x, W, b):
    return matmul(x, W) + b

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)