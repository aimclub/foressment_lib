import unittest
from aopssop import aossop


class AossopTestCase(unittest.TestCase):

    def test_smth(self):
        result = aossop('Hello World!')
        self.assertEqual(result, 'smth')
