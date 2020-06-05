import unittest
from linear_algebra.linear_algebra_core import *


class LinearAlgebraTest(unittest.TestCase):

    def test_matrix(self):
        def_matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        def_vector = Vector([1, 2, 3])

        # Matrix construction
        self.assertEqual(Matrix([[1, 1], [1, 1]]).data, [[1, 1], [1, 1]])

        # Matrix functions
        self.assertEqual(Matrix([[1, 2], [3, 4]]).transpose(), Matrix([[1, 3], [2, 4]]))
        self.assertEqual(Matrix([[1, 2], [3, 4], [5, 6]]).transpose(), Matrix([[1, 3, 5], [2, 4, 6]]))
        self.assertEqual(Matrix([[1, 0], [0, 1]]).determinant(), 1)
        self.assertEqual(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to_echelon(),
                         Matrix([[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]]))
        self.assertEqual((def_matrix.add_col(def_vector)).n_cols, 4)
        self.assertEqual((def_matrix.add_col(def_matrix)).n_cols, 6)
        self.assertEqual((def_matrix.add_row(def_vector.transpose())).n_rows, 4)
        self.assertEqual((def_matrix.add_row(def_matrix)).n_rows, 6)
        self.assertEqual(def_matrix.row(0), def_vector.transpose())
        self.assertEqual(def_matrix.col(0), Vector([1, 4, 7]))

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
    
    matrix = [[10, 4, 6, 12, 7, 1, 0],
              [5, 8, 11, 9, 13, 0, 0],
              [14, 3, 15, 16, 17, 0, 1],
              [18, 2, 19, 20, 21, 0, 2],
              [22, 23, 24, 25, 26, 0, 0],
              [3, 4, 5, 5, 4, 3, 0],
              [3, 1, 14, 1, 1, 1, 5]]
    ma = Matrix(matrix)
    ma_inv = ma.inverse()

    print(solve_linear_system(
        Matrix([[1, 2, 3], [2, 0, 2], [0, 1, -2]]),
        Vector([1, 2, 3])))

    print(solve_linear_system(ma,
                              Vector([1, 10, -3, 5, 3, 4, 8])))
