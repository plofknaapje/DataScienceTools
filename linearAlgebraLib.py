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

    def to_echelon(self):
        echelon = self.data
        # TODO for col in range(ncols):

    def determinant(self):
        if self.nrows != self.ncols:
            raise Exception('Matrix is not square') from None
        if self.nrows == 2:
            return (self.data[0][0] * self.data[1][1]) - (self.data[1][0] * self.data[0][1])
        if self.nrows == 3:
            return 1

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
        Equality of vectors
        :param other: Vector
        :return: Boolean
        """
        return self.data == other.data

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
        Multiplies a vector with a scalar or a vector
        :param other: Either a scalar or a vector
        :return: Either a matrix (Mx1 * 1xN), a vector (Mx1 * 1) or a scalar (1xM * Mx1)
        """
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return Vector([i * other for i in self.data], self.column)
        elif isinstance(other, Vector):
            len_other = len(other)
            if len(self) == len_other and not self.column and other.column:
                return sum([self.data[i] * other.data[i] for i in range(len(self))])
            elif self.column and not other.column:
                return Matrix([[r * c for c in other.data] for r in self.data])
            elif len(self) == len_other and (self.column == other.column):
                raise Exception('Cant multiply vectors of same length and orientation')
            else:
                raise Exception('Vectors are not compatible for multiplication')
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


def vec_scale_div(vec, scalar):
    return [i / scalar for i in vec]


def identity_matrix(n):
    data = [[1 if c == r else 0 for c in range(n)] for r in range(n)]
    return Matrix(data)
