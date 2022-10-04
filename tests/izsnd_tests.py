import unittest
from aopssop import izsnd


class IzsndTestCase(unittest.TestCase):

    def test_smth(self):
        result = izsnd('Hello World!')
        self.assertEqual(result, 'smth')
