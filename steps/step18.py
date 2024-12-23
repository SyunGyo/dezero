from src.chap02 import square, Variable, add
import numpy as np

def main():
    x0 = Variable(np.array(3.0))
    x1 = Variable(np.array(5.0))

    t = add(x0, x1)
    y = add(t,x0)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

main()