import unittest
from aopssop import apssop


class ApssopTestCase(unittest.TestCase):

    def test_smth(self):
        result = apssop('Hello World!')
        self.assertEqual(result, 'smth')
