from src.chap02 import square, Variable, add, mul
import numpy as np

def main():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    

    y = 2.0 * a + 1.0
    y.backward()

    print(y)
    print(a.grad)
    # print(b.grad)

main()