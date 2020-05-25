import math
from fractions import Fraction
from timeit import timeit

from helpers.matrix_helpers import *


class Matrix:

    def __init__(self, data):
        """
        :param data: a list with lists of equal length which only contain numbers
        """
        row_len = len(data[1])
        for row in data:
            if len(row) != row_len:
                raise Exception('Rows not equal length') from None
            for point in row:
                if not (isinstance(point, (int, float, complex, Fraction)) and not isinstance(point, bool)):
                    raise Exception('Value is not a number') from None
        self.data = [[Fraction(col) for col in row] for row in data]
        self.n_rows = len(data)
        self.n_cols = len(data[1])

    def __eq__(self, other):
        return self.data == other.data

    def __mul__(self, other):
        # other is a scalar
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return Matrix([[p * other for p in row] for row in self.data])
        # other is Vector
        elif isinstance(other, Vector):
            if other.column:
                if len(other) == self.n_cols:
                    other = other.transpose()
                    return Vector([other * Vector(row) for row in self.data])
                else:
                    raise ValueError('Dimensions of Matrix and Vector do not match')
            else:
                raise ValueError('Matrix can only be multiplied with a column Vector')
        elif isinstance(other, Matrix):
            if self.n_cols == other.n_rows:
                other = other.transpose()
                return Matrix([[Vector(row, False) * Vector(col) for col in other.data] for row in self.data])
            else:
                raise ValueError('Dimensions of matrices do not match')
        else:
            raise TypeError('Type is not compatible with matrix multiplication')

    def __truediv__(self, other):
        # other is scalar
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return [[p / other for p in row] for row in self.data]
        else:
            raise TypeError('Matrix can only be divided by a scalar')

    def __add__(self, other):
        # other is scalar
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return [[p + other for p in row] for row in self.data]
        # other is Matrix
        elif isinstance(other, Matrix):
            if self.n_cols == other.n_cols and self.n_rows == other.n_rows:
                return Matrix([[self.data[row][col] + other.data[row][col]
                                for col in range(self.n_cols)] for row in range(self.n_rows)])
            else:
                raise ValueError('Dimensions of matrices do not match')
        else:
            raise TypeError('Matrix can only be divided by a scalar')

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __str__(self):
        rows = ['[' + ', '.join([str(i) for i in row]) + ']' for row in self.data]
        return '\n'.join(rows)

    def transpose(self):
        return Matrix([[self.data[r][c] for r in range(len(self.data))]
                       for c in range(len(self.data[1]))])

    def to_echelon(self, report_ops=False):
        matrix, ops = reduce_to_echelon(self.data.copy(), True)
        if report_ops:
            return Matrix(matrix), ops
        else:
            return Matrix(matrix)

    def to_reduced_echelon(self, report_ops=False):
        matrix, ops = reduce_to_red_echelon(self.data.copy(), True)
        if report_ops:
            return Matrix(matrix), ops
        else:
            return Matrix(matrix)

    def determinant(self):
        if self.n_rows != self.n_cols:
            raise Exception('Matrix is not square')
        if self.n_rows == 2:
            return (self.data[0][0] * self.data[1][1]) - (self.data[1][0] * self.data[0][1])
        else:
            echelon, ops = reduce_to_echelon(self.data.copy(), True)
            swaps = sum([1 if row[0] == 'swap' else 0 for row in ops])
            return math.prod([echelon[i][i] for i in range(len(echelon))]) * (-1) ** swaps

    def inverse(self):
        if self.determinant() != 0:
            ops = reduce_to_red_echelon(self.data.copy(), True)[1]
            matrix = identity_matrix(self.n_rows).data

            if isinstance(ops[0], str):
                ops = [ops]

            for op in ops:
                if op[0] == 'swap':
                    matrix = row_swap(matrix, op[1], op[2])
                elif op[0] == 'multiplication':
                    matrix = row_multiply(matrix, op[1], op[2])
                elif op[0] == 'subtract':
                    matrix = row_subtract(matrix, op[1], op[2], op[3])
                else:
                    raise ValueError('Row operation not recognized')
        else:
            raise ValueError('Matrix has a determinant of 0 and is not invertible')
        return Matrix(matrix)


class Vector:

    def __init__(self, lst, column=True):
        """
        :param lst: list of numbers
        :param column: True if column vector
        """
        for point in lst:
            if not (isinstance(point, (int, float, complex, Fraction)) and not isinstance(point, bool)):
                raise TypeError('Value is not a number')
        self.data = [Fraction(i) for i in lst]
        self.column = column

    def __len__(self):
        """
        Length of vector
        :return: scalar
        """
        return len(self.data)

    def __eq__(self, other):
        """
        Comparison of vectors
        :param other: Vector
        :return: Boolean
        """
        return self.data == other.data and self.column == other.column

    def __add__(self, other):
        """
        Add scalar or Vector to Vector
        :param other: scalar or Vector
        :return: Vector
        """
        # other is a scalar
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return Vector([i + other for i in self.data], self.column)
        # other is a Vector
        elif isinstance(other, Vector):
            if len(self.data) != len(other):
                raise Exception('Vectors are not of equal length')
            elif self.column != other.column:
                raise Exception('Vectors are not of equal orientation')
            else:
                return Vector([self.data[i] + other.data[i] for i in range(len(self.data))], self.column)
        # other is not a scalar or a Vector
        else:
            raise Exception('Argument is not a number or a Vector') from TypeError

    def __mul__(self, other):
        """
        Multiplies a vector with a scalar, Matrix or Vector
        :param other: Either a scalar, Matrix or Vector
        :return: Either a Matrix (Mx1 * 1xN), a Vector (Mx1 * 1), a Vector (1xM * MxN) or a Scalar (1xM * Mx1)
        """
        # other is a number
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return Vector([i * other for i in self.data], self.column)
        # other is a Vector
        elif isinstance(other, Vector):
            len_other = len(other)
            # Lengths are the same and self is row and other is col
            if len(self) == len_other and not self.column and other.column:
                return sum([self.data[i] * other.data[i] for i in range(len(self))])
            # Self is col and other is row
            elif self.column and not other.column:
                return Matrix([[row * col for col in other.data] for row in self.data])
            elif len(self) == len_other and (self.column == other.column):
                raise Exception('Cant multiply vectors of same length and orientation')
            else:
                raise Exception('Vectors are not compatible for multiplication')
        # other is a Matrix
        elif isinstance(other, Matrix):
            if not self.column and other.n_rows == len(self):
                return Vector([sum([self.data[r] * other.data[r][c] for r in range(other.n_rows)])
                               for c in range(other.n_cols)], column=False)
            elif self.column:
                raise Exception('Column Vector cant be multiplied by a Matrix')
            else:
                raise Exception('Dimensions of Vector and Matrix are not compatible')
        # other is not a scalar, Vector or Matrix
        else:
            raise TypeError('Argument is not a number or a Vector')

    def __truediv__(self, other):
        """
        Divides a vector by a scalar
        :param other: scalar
        :return: vector
        """
        # other is a scalar
        if isinstance(other, (int, float, complex, Fraction)) and not isinstance(other, bool):
            return Vector([i / other for i in self.data], self.column)
        # other is not a scalar
        else:
            raise TypeError('Argument is not a number')

    def __sub__(self, other):
        """
        Subtracts a vector or scalar from a vector
        :param other: scalar or vector
        :return: vector
        """
        return self.__add__(other * -1)

    def __str__(self):
        lst = [str(i) for i in self.data]
        if self.column:
            return '[' + ', '.join(lst) + ']\''
        else:
            return '[' + ', '.join(lst) + ']'

    def transpose(self):
        """
        Transpose vector
        :return: vector
        """
        return Vector(self.data, not self.column)


def identity_matrix(n):
    """
    Returns a Matrix of size n*n with 1's on the diagonal
    :param n: Matrix row and col size
    :return: nxn Matrix
    """
    data = [[1 if c == r else 0 for c in range(n)] for r in range(n)]
    return Matrix(data)


if __name__ == '__main__':
    matrix = [[10, 4, 6, 12, 7, 1, 0],
      [5, 8, 11, 9, 13, 0, 0],
      [14, 3, 15, 16, 17, 0, 1],
      [18, 2, 19, 20, 21, 0, 2],
      [22, 23, 24, 25, 26, 0, 0],
      [3, 4, 5, 5, 4, 3, 0],
      [3, 1, 14, 1, 1, 1, 5]]
    ma = Matrix(matrix)
    ma_inv = ma.inverse()


    def create_matrix():
        return Matrix(matrix)


    def matrix_inverse():
        return Matrix(matrix).inverse()
    
    def matrix_mult():
        return ma*ma_inv


    print(timeit(create_matrix, number=5)/5)
    print(ma, '\n')

    print(timeit(matrix_inverse, number=5)/5)
    print(ma_inv, '\n')

    print(timeit(matrix_mult, number=5)/5)
    print(ma * ma_inv)
