from src.chap02 import Variable, add
import numpy as np

def main():
    x = Variable(np.array(3.0))
    y = add(x,x)

    print('y', y.data)
    y.backward()
    print('x.grad', x.grad)

    x.cleargrad()
    y = add(add(x,x),x)
    y.backward()
    print('x.grad', x.grad)
main()