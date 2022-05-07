import unittest

from .. import imagenn

class MyTestCase(unittest.TestCase):
    def test_something(self):
        print("hi")

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
