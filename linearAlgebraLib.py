import math

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
                if not (isinstance(point, (int, float, complex)) and not isinstance(point, bool)):
                    raise Exception('Value is not a number') from None
        self.data = data
        self.nrows = len(data)
        self.ncols = len(data[1])

    def __eq__(self, other):
        return self.data == other.data

    def transpose(self):
        return Matrix([[self.data[r][c] for r in range(len(self.data))]
                       for c in range(len(self.data[1]))])

    def to_echelon(self, report_ops=False):
        return Matrix(reduce_echelon(self.data, report_ops))

    def determinant(self):
        if self.nrows != self.ncols:
            raise Exception('Matrix is not square')
        if self.nrows == 2:
            return (self.data[0][0] * self.data[1][1]) - (self.data[1][0] * self.data[0][1])
        else:
            echelon, ops = self.to_echelon(True)
            swaps = sum([1 if row[0] == 'swap' else 0 for row in ops])
            return math.prod([echelon[i][i] for i in range(len(echelon))]) * (-1)**swaps
            

    # TODO def inverse(self):


class Vector:

    def __init__(self, lst, column=True):
        """
        :param lst: list of numbers
        :param column: True if column vector
        """
        for point in lst:
            if not (isinstance(point, (int, float, complex)) and not isinstance(point, bool)):
                raise TypeError('Value is not a number')
        self.data = lst
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
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return Vector([i + other for i in self.data], self.column)
        elif isinstance(other, Vector):
            if len(self.data) != len(other):
                raise Exception('Vectors are not of equal length')
            elif self.column != other.column:
                raise Exception('Vectors are not of equal orientation')
            else:
                return Vector([self.data[i] + other.data[i] for i in range(len(self.data))], self.column)
        else:
            raise Exception('Argument is not a number or a Vector') from TypeError

    def __mul__(self, other):
        """
        Multiplies a vector with a scalar, Matrix or Vector
        :param other: Either a scalar, Matrix or Vector
        :return: Either a Matrix (Mx1 * 1xN), a Vector (Mx1 * 1), a Vector (1xM * MxN) or a Scalar (1xM * Mx1)
        """
        # other is a number
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return Vector([i * other for i in self.data], self.column)
        # other is a Vector
        elif isinstance(other, Vector):
            len_other = len(other)
            # Lenths are the same and self is row and other is col
            if len(self) == len_other and not self.column and other.column:
                return sum([self.data[i] * other.data[i] for i in range(len(self))])
            # Self is col and other is row
            elif self.column and not other.column:
                return Matrix([[r * c for c in other.data] for r in self.data])
            elif len(self) == len_other and (self.column == other.column):
                raise Exception('Cant multiply vectors of same length and orientation')
            else:
                raise Exception('Vectors are not compatible for multiplication')
        # other is a Matrix    
        elif isinstance(other, Matrix):
            if not self.column and other.nrows == len(self):
                return Vector([sum([self.data[r] * other.data[r][c] for r in range(other.nrows)]) 
                               for c in range(other.ncols)], column=False)
            elif self.column:
                raise Exception('Column Vector cant be multiplied by a Matrix')
            else:
                raise Exception('Dimensions of Vector and Matrix are not compatible')
        else:
            raise TypeError('Argument is not a number or a Vector')

    def __truediv__(self, other):
        """
        Divides a vector by a scalar
        :param other: scalar
        :return: vector
        """
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return Vector([i / other for i in self.data], self.column)
        else:
            raise TypeError('Argument is not a number')

    def __sub__(self, other):
        """
        Subtracts a vector or scalar from a vector
        :param other: scalar or vector
        :return: vector
        """
        return self.__add__(other * -1)

    def transpose(self):
        """
        Transpose vector
        :return: vector
        """
        return Vector(self.data, not self.column)


def ma_reduce_col(data, col):
    if col != 0:
        for row_num, row in enumerate(data):
            if row[col - 1] == 0:
                start_row = row_num
    else:
        start_row = 0

    if start_row + 1 == len(data):
        return data

    for i, row in enumerate(data[start_row + 1:]):
        factor = row[col] / data[start_row][col]
        data[i] = row


def identity_matrix(n):
    data = [[1 if c == r else 0 for c in range(n)] for r in range(n)]
    return Matrix(data)


def row_swap(matrix, row1, row2):
    if row1 != row2:
        matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
        return matrix
    elif row1 == row2:
        return matrix


def subtract_row(matrix, row1, row2, factor):
    matrix[row2] = [matrix[row2][i] - (matrix[row1][i] * factor) for i in range(len(matrix[0]))]
    return matrix


def reduce_col(matrix, col, return_ops=False):
    ops = []
    current_row = 0
    # Special treatment for first column
    if col == 0:
        if matrix[0][0] == 0:
            for i in range(len(matrix)):
                if matrix[i][0] != 0:
                    matrix = row_swap(matrix, 0, i)
                    ops.append(['swap', 0, i])
                    break
    # General starting row detection
    else:
        for i in range(len(matrix) - 2, -1, -1):
            if matrix[i][col - 1] != 0:
                current_row = i + 1
                break
        if matrix[current_row][col] == 0:
            for i in range(current_row + 1, len(matrix)):
                if matrix[i][col] != 0:
                    matrix = row_swap(matrix, current_row, i)
                    ops.append(['swap', current_row, i])
                    break
    # If starting row number not 0, then start subtracting
    if matrix[current_row][col] != 0 and current_row != len(matrix):
        for row in range(current_row + 1, len(matrix)):
            if matrix[row][col] != 0:
                factor = matrix[row][col] / matrix[current_row][col]
                matrix = subtract_row(matrix, current_row, row, factor)
                ops.append(['subtract', current_row, row, factor])
    if return_ops:
        return matrix, ops
    else:
        return matrix


def clean_ops(ops):
    lst = []
    for set in ops:
        for op in set:
            lst.append(op)
    return lst


def reduce_echelon(matrix, report_ops=False):
    ops = []
    for i in range(len(matrix[0])):
        if report_ops:
            matrix, ops_update = reduce_col(matrix, i, report_ops)
            if ops_update != []:
                ops.append(ops_update)
        else:
            matrix = reduce_col(matrix, i, report_ops)
    if report_ops:
        return matrix, clean_ops(ops)
    else:
        return matrix

if __name__ == '__main__':
    #print(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to_echelon().data)
    #print(Matrix([[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]]).data)
