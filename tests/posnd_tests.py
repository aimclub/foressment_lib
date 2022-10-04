import unittest
from aopssop import posnd


class PosndTestCase(unittest.TestCase):

    def test_smth(self):
        result = posnd('Hello World!')
        self.assertEqual(result, 'smth')
