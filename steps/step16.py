from src.chap02 import square, Variable, add
import numpy as np

def main():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print('y', y.data)
    print('x.grad', x.grad )

main()