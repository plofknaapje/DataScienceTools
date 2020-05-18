import unittest
import linearAlgebraLib as la


class LinearAlgebraTest(unittest.TestCase):

    def test_matrix(self):
        # Matrix construction
        self.assertEqual(la.Matrix([[1, 1], [1, 1]]).data, [[1, 1], [1, 1]])

        # Matrix functions
        self.assertEqual(la.Matrix([[1, 2], [3, 4]]).transpose(), la.Matrix([[1, 3], [2, 4]]))
        self.assertEqual(la.Matrix([[1, 2], [3, 4], [5, 6]]).transpose(), la.Matrix([[1, 3, 5], [2, 4, 6]]))
        self.assertEqual(la.Matrix([[1, 0], [0, 1]]).determinant(), 1)
        self.assertEqual(la.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to_echelon(), 
                         la.Matrix([[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]]))

    def test_identity(self):
        self.assertEqual(la.identity_matrix(2).data, [[1, 0], [0, 1]])

    def test_vector(self):
        # Vector construction
        self.assertEqual(la.Vector([1, 2, 3, 4]).data, [1, 2, 3, 4])
        with self.assertRaises(TypeError):
            temp = la.Vector([1, 2, '3', 4])

        # Vector functions
        self.assertEqual(len(la.Vector([1, 2, 3])), 3)
        self.assertFalse(la.Vector([1, 2, 3]).transpose().column)

        # Vector-scalar functions
        self.assertEqual((la.Vector([1, 2, 3]) + 3), la.Vector([4, 5, 6]))
        self.assertEqual((la.Vector([3, 4, 5]) - 2), la.Vector([1, 2, 3]))
        self.assertEqual((la.Vector([2, 2, 2]) * 2), la.Vector([4, 4, 4]))
        self.assertEqual((la.Vector([2, 2, 2]) / 2), la.Vector([1, 1, 1]))

        # Vector-vector functions
        self.assertEqual((la.Vector([1, 2, 3]) + la.Vector([1, 2, 3])), la.Vector([2, 4, 6]))
        self.assertEqual((la.Vector([1, 2, 3]) - la.Vector([1, 2, 3])), la.Vector([0, 0, 0]))
        self.assertEqual((la.Vector([1, 1, 1], False) * la.Vector([1, 1, 1])), 3)
        self.assertEqual((la.Vector([1, 1, 1]) * la.Vector([1, 1, 1], False)),
                         la.Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        
        # Vector-matrix functions
        self.assertEqual((la.Vector([1, 1, 1], False) * la.Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])), 
                         la.Vector([3, 3, 3], False))


if __name__ == '__main__':
    unittest.main()
