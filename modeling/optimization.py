import linear_algebra.linear_algebra_core as core

class LinearModel:
    
    def __init__(self, form='standard'):
        self.form = form
        
    def solve(self, matrix_a, c, b, signs=[], x_range=[], goal='min'):
        if self.form == 'standard':
            return 1
            # TODO
        else:
            raise Exception('Only standard form problems have been implemented')
        
        
def simplex(bfs, basis, matrix_a, c):
    non_basic = [i for i in range(matrix_a.n_rows) if i not in basis]
    basic_matrix = matrix_a.cols(basis)
    c_b = core.Vector([c.data[i] for i in basis], False)
    a_non_basic = matrix_a.cols(non_basic)
    c_j = c - (c_b * basic_matrix.inverse() * matrix_a).transpose()
    
    for i in c_j.data:
        if i <= 0:
            new_basic = i
            break
    
    
            
    return c_j

if __name__ == "__main__":
    c = core.Vector([-1, 1, -1, 0, 0, 0])
    a = core.Matrix([[3, 2, 2, 1, 0, 0],
                     [1, -2, 4, 0, 1, 0],
                     [1, 2, -2, 0, 0, 1]])
    b = core.Vector([120, 20, 40])
    basis = [3, 4, 5]
    bfs = core.Vector([0, 0, 0, 120, 20 , 40])
    
    print(simplex(bfs, basis, a, c))
