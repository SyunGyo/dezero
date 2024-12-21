import unittest
import numpy as np
from steps.step02 import Variable, square, add

class AddTest(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        c = add(a,b)

        self.assertEqual(c.data, 5)
        
