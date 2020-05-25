import unittest
from linearAlgebraLib import *


class LinearAlgebraTest(unittest.TestCase):

    def test_matrix(self):
        # Matrix construction
        self.assertEqual(Matrix([[1, 1], [1, 1]]).data, [[1, 1], [1, 1]])

        # Matrix functions
        self.assertEqual(Matrix([[1, 2], [3, 4]]).transpose(), Matrix([[1, 3], [2, 4]]))
        self.assertEqual(Matrix([[1, 2], [3, 4], [5, 6]]).transpose(), Matrix([[1, 3, 5], [2, 4, 6]]))
        self.assertEqual(Matrix([[1, 0], [0, 1]]).determinant(), 1)
        self.assertEqual(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to_echelon(),
                         Matrix([[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]]))

        # Inverse
        matrix = Matrix([[2, 0, 1], [1, 1, -4], [3, 7, -3]])
        matrix_inv = matrix.inverse()
        self.assertEqual(matrix * matrix_inv, identity_matrix(3))

    def test_identity(self):
        self.assertEqual(identity_matrix(2).data, [[1, 0], [0, 1]])

    def test_vector(self):
        # Vector construction
        self.assertEqual(Vector([1, 2, 3, 4]).data, [1, 2, 3, 4])
        with self.assertRaises(TypeError):
            temp = Vector([1, 2, '3', 4])

        # Vector functions
        self.assertEqual(len(Vector([1, 2, 3])), 3)
        self.assertFalse(Vector([1, 2, 3]).transpose().column)

        # Vector-scalar functions
        self.assertEqual((Vector([1, 2, 3]) + 3), Vector([4, 5, 6]))
        self.assertEqual((Vector([3, 4, 5]) - 2), Vector([1, 2, 3]))
        self.assertEqual((Vector([2, 2, 2]) * 2), Vector([4, 4, 4]))
        self.assertEqual((Vector([2, 2, 2]) / 2), Vector([1, 1, 1]))

        # Vector-vector functions
        self.assertEqual((Vector([1, 2, 3]) + Vector([1, 2, 3])), Vector([2, 4, 6]))
        self.assertEqual((Vector([1, 2, 3]) - Vector([1, 2, 3])), Vector([0, 0, 0]))
        self.assertEqual((Vector([1, 1, 1], False) * Vector([1, 1, 1])), 3)
        self.assertEqual((Vector([1, 1, 1]) * Vector([1, 1, 1], False)),
                         Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        
        # Vector-matrix functions
        self.assertEqual((Vector([1, 1, 1], False) * Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])), 
                         Vector([3, 3, 3], False))


if __name__ == '__main__':
    unittest.main()
