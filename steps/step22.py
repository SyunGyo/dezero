from src.chap02 import square, Variable, add, mul
import numpy as np

def main():
    x = Variable(np.array(2.0))
    y = x ** 3
    y.backward()
    print(y)
    print(x.grad)
    

main()