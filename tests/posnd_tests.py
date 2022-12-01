import unittest
import pandas as pd
import numpy as np

from aopssop import \
    CheckDataTypes, ClusterFilling, Multicolinear, Informativity, __Verbose__, Data


class PosndTestCase(unittest.TestCase):

    def test_smth(self):
        result = 'Hello World!'
        self.assertEqual(result, 'smth')
