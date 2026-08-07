import sys
import unittest


class PythonVersionTests(unittest.TestCase):
    def test_python_312_is_required(self):
        self.assertEqual(sys.version_info[:2], (3, 12))


if __name__ == "__main__":
    unittest.main()
