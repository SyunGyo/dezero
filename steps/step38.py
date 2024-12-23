import numpy as np 
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = x.T
y.backward()
print(y)
print(x.grad)