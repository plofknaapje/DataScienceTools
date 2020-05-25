

def row_swap(matrix, row1, row2):
    """
    Swaps two rows in a matrix
    :param matrix: list of lists
    :param row1: number of a row
    :param row2: number of a row
    :return: a list of lists
    """
    if row1 != row2:
        matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
        return matrix
    elif row1 == row2:
        return matrix


def row_subtract(matrix, row1, row2, factor):
    """
    Subtracts a multiple of one row from another
    :param matrix: list of lists of equal length containing numbers
    :param row1: index of a row
    :param row2: index of a row
    :param factor: multiplication factor of row1
    :return: list of lists of equal length containing numbers
    """
    matrix[row2] = [matrix[row2][i] - (matrix[row1][i] * factor) for i in range(len(matrix[0]))]
    return matrix


def row_multiply(matrix, row, factor):
    """
    Multiplies a row by a factor
    :param matrix: List of lists of equal length containing numbers
    :param row: index of a row
    :param factor: multiplying factor
    :return: List of lists of equal length containing numbers
    """
    matrix[row] = [i*factor for i in matrix[row]]
    return matrix


def col_reduce(matrix, col, return_ops=False):
    """
    Reduces column down into echelon form by transforming all numbers below the pivot position into 0's
    :param matrix: list of lists of equal length containing numbers
    :param col: index of column
    :param return_ops: performed operations are returned
    :return: list of lists of equal length containing numbers
    """
    ops = []
    pivot_row = len(matrix)  # Default to last row

    # Special treatment for first column to find the pivot row
    if col == 0:
        if matrix[0][0] == 0:
            for i in range(len(matrix)):
                if matrix[i][0] != 0:
                    matrix = row_swap(matrix, 0, i)
                    ops.append(['swap', 0, i])
                    break
        pivot_row = 0

    # General pivot row detection
    else:
        for i in range(len(matrix) - 2, -1, -1):
            if matrix[i][col - 1] != 0:
                pivot_row = i + 1
                break
        if matrix[pivot_row][col] == 0:
            for i in range(pivot_row + 1, len(matrix)):
                if matrix[i][col] != 0:
                    matrix = row_swap(matrix, pivot_row, i)
                    ops.append(['swap', pivot_row, i])
                    break

    # With a non-zero number in the pivot position, transform all numbers below to 0
    if matrix[pivot_row][col] != 0 and pivot_row != len(matrix):
        for row in range(pivot_row + 1, len(matrix)):
            if matrix[row][col] != 0:
                factor = matrix[row][col] / matrix[pivot_row][col]
                matrix = row_subtract(matrix, pivot_row, row, factor)
                ops.append(['subtract', pivot_row, row, factor])

    if return_ops:
        return matrix, ops
    else:
        return matrix


def col_rev_reduce(matrix, col, return_ops=False):
    """
    Reduces a column into reduced echelon form by transforming all numbers above the pivot position into 0's
    :param matrix: list of lists of equal length containing numbers
    :param col: index of column
    :param return_ops: performed operations are returned
    :return: list of lists of equal length containing numbers
    """
    ops = []
    pivot_row = 0  # Defaults to top row
    
    # Find pivot row of the column
    for row in range(len(matrix)-1, -1, -1):
        if matrix[row][col] != 0:
            pivot_row = row
            break
    
    # Transform all numbers above the pivot to 0
    if matrix[pivot_row][col] != 0 and matrix[pivot_row][col] != 1:
        factor = 1 / matrix[pivot_row][col]
        matrix = row_multiply(matrix, pivot_row, factor)
        ops.append(['multiplication', pivot_row, factor])
    if pivot_row != 0:
        for row in range(pivot_row):
            if matrix[row][col] != 0:
                factor = matrix[row][col] / matrix[pivot_row][col]
                matrix = row_subtract(matrix, pivot_row, row, factor)
                ops.append(['subtract', pivot_row, row, factor])

    if return_ops:
        return matrix, ops
    else:
        return matrix


def clean_ops(ops):
    cleaned_ops = []
    for lst in ops:
        if isinstance(lst[0], str):
            cleaned_ops.append(lst)
        else:
            for op in lst:
                cleaned_ops.append(op)

    return cleaned_ops


def reduce_to_echelon(matrix, return_ops=False):
    ops = []

    for col in range(len(matrix[0])):
        matrix, ops_update = col_reduce(matrix, col, True)
        if ops_update:
            ops.append(ops_update)

    if return_ops:
        return matrix, clean_ops(ops)
    else:
        return matrix


def reduce_to_red_echelon(matrix, return_ops=False):
    matrix, ops = reduce_to_echelon(matrix, True)

    for col in range(len(matrix[0])):
        matrix, ops_update = col_rev_reduce(matrix, col, True)
        if ops_update:
            ops.append(ops_update)

    if return_ops:
        return matrix, clean_ops(ops)
    else:
        return matrix
