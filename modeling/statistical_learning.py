import linear_algebra.linear_algebra_core as la


class Model:

    def fit(self, x, y):
        raise Exception('Fitting not implemented')

    def predict(self, x):
        raise Exception('Predicting not implemented')

    def evaluate(self, x, y, metrics=[]):
        raise Exception('Evaluation not implemented')

    def set_params(self, parameters):
        raise Exception('Not implemented')

    def get_params(self):
        raise Exception('Not implemented')


class LinearRegression(Model):

    def __init__(self, criterion='LS'):
        self.criterion = criterion
        self.coefficients = None

    def fit(self, x, y):
        """
        Fits the parameters of X to predict the value y with model criterion
        :param x: Matrix with equal amounts of rows as y
        :param y: Column Vector with length equal to rows in X
        """
        if self.criterion != 'LS':
            raise Exception('Only Least Squares is implemented')
        regression_data_check(x, y)
        self.coefficients = (x.transpose() * x).inverse() * x.transpose() * y

    def predict(self, x):
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        regression_data_check(x, width=len(self.coefficients))
        return x * self.coefficients


def regression_data_check(x, y=None, width=None):
    if not isinstance(x, la.Matrix):
        raise TypeError('x has to be a Matrix')

    if y is not None:
        if not isinstance(y, la.Vector):
            raise TypeError('y has to be a Vector')
        if not y.column:
            raise ValueError('Only column Vectors are accepted')
        if len(y) != x.n_rows:
            raise ValueError('Length of y has to be the same as number of rows'
                             'in x')

    if width is not None:
        if x.n_cols != width:
            raise ValueError('Number of cols in x has to be equal to width')
        
if __name__ == "__main__":
    x = la.Matrix([[1, 2], [2, 2], [3, 1]])
    y = la.Vector([4, 6, 7])
    model = LinearRegression()
    model.fit(x, y)
    print(model.coefficients)
    print(model.predict(la.Matrix([[2, 3], [3, 3]])))
