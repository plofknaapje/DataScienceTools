matrix1 = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

matrix2 = [[0, 2, 3],
           [1, 4, 5],
           [1, 2, 0]]

matrix3 = [[0, 1, 1],
           [0, 0, 2],
           [1, 2, 3]]

matrix4 = [[1, 2, 3, 4],
           [2, 3, 4, 5],
           [0, 5, 5, 5]]


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
        matrix, ops_update = reduce_col(matrix, i, report_ops)
        if ops_update != []:
            ops.append(ops_update)
    if report_ops:
        return matrix, clean_ops(ops)
    else:
        return matrix


# print(row_swap(matrix1, 1, 2))
# print(subtract_row(matrix1, 0, 1, 4))
# print(reduce_col(matrix3, 0, False))
# print(reduce_col(matrix2, 0))

print(reduce_echelon(matrix1, True))
