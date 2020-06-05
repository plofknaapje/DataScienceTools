import linear_algebra.linear_algebra_core as core
import modeling.metrics.regression_metrics as reg_met
import modeling.evaluation.regression_evaluation as reg_eval


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

    def __init__(self, criterion='LS', intercept=True):
        self.criterion = criterion
        self.intercept = intercept
        self.coefficients = None

    def fit(self, x, y):
        """
        Fits the parameters of X to predict the value y with model criterion
        :param x: Matrix with equal amounts of rows as y
        :param y: Column Vector with length equal to rows in X
        :param intercept: True if intercept has to be added to x
        """
        if self.criterion != 'LS':
            raise Exception('Only Least Squares is implemented')
        regression_data_check(x, y)
        
        if self.intercept:
            x = core.Matrix([[1] + row for row in x.data])
            
        self.coefficients = (x.transpose() * x).inverse() * x.transpose() * y

    def predict(self, x):
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        
        if self.intercept:
            regression_data_check(x, width=len(self.coefficients) - 1)
            x = core.Matrix([[1] + row for row in x.data])
        else:
            regression_data_check(x, width=len(self.coefficients))
            
        return x * self.coefficients

    def score(self, x, y, metric=reg_met.r_squared, number_type=float):
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        return metric(y, self.predict(x), number_type)

    def get_params(self):
        return self.criterion

    def set_params(self, criterion):
        self.criterion = criterion


def regression_data_check(x, y=None, width=None):
    if not isinstance(x, core.Matrix):
        raise TypeError('x has to be a Matrix')

    if y is not None:
        if not isinstance(y, core.Vector):
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
    x = core.Matrix([[1, 2], [2, 2], [3, 1], [4, 2]])
    y = core.Vector([4, 6, 6.5, 9])
    model = LinearRegression()
    model.fit(x, y)
    print(model.coefficients)
    print(model.predict(core.Matrix([[2, 3], [3, 3]])))
    print(model.score(x, y))
    print(reg_eval.correlation(x.col(0), y))
