import linear_algebra.linear_algebra_core as core
import modeling.metrics.regression_metrics as reg_met
import modeling.evaluation.regression_evaluation as reg_eval
from tabulate import tabulate
from scipy.stats import f, t


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
        x = core.enhance_matrix(x)

        self.coefficients = (x.transpose() * x).inverse() * x.transpose() * y

    def predict(self, x):
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        x = core.enhance_matrix(x)
        regression_data_check(x, width=len(self.coefficients))

        return x * self.coefficients

    def score(self, x, y_true, metric=reg_met.r_squared, number_type=float):
        x = core.enhance_matrix(x)
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        return metric(y_true, self.predict(x), number_type)

    def evaluate(self, x, y_true):
        """
        Evaluates the performance of the trained model on a global and variable
        level. For global, RSE, R^2 and F-statistic are standard. For variables
        the SE and t-statistic is used.
        :param x: Matrix of predictors
        :param y_true: Vector of true y values
        :return: 
        """
        x = core.enhance_matrix(x)
        y_pred = self.predict(x)
        global_metrics = [['RSE', reg_eval.residual_standard_error],
                          ['R^2', reg_met.r_squared],
                          ['F-statistic', reg_eval.f_statistic],
                          ['p-value']]
        var_metrics = [['SE', reg_eval.standard_error_coefs],
                       ['t-statistic', reg_eval.t_statistic],
                       ['p-value']]

        glob_outcomes = {'Metric': [], 'Value': []}
        for i in global_metrics:
            if len(i) > 1:
                glob_outcomes['Metric'].append(i[0])
                glob_outcomes['Value'].append(i[1](x=x, y_true=y_true, y_pred=y_pred,
                                                   num_predictors=x.n_cols))
            elif i[0] == 'p-value':
                glob_outcomes['Metric'].append(i[0])
                glob_outcomes['Value'].append(f.sf(glob_outcomes['Value'][2],
                                              dfn=len(y_pred), dfd=x.n_cols - 1))
            else:
                raise Exception('Single value metric not implemented')

        var_outcomes = {'Column': list(range(x.n_cols)),
                        'Coefficient': self.coefficients.data}
        for i in var_metrics:
            if len(i) > 1:
                var_outcomes[i[0]] = i[1](x=x, y_true=y_true, y_pred=y_pred,
                                                       coefs=var_outcomes['Coefficient'])
            elif i[0] == 'p-value':
                var_outcomes[i[0]] = [2*t.sf(abs(float(score)), len(y_pred) - x.n_cols)
                                      for score in var_outcomes['t-statistic']]

        print(tabulate(glob_outcomes, headers='keys'))
        print(tabulate(var_outcomes, headers='keys'))

        return glob_outcomes, var_outcomes

    def get_params(self):
        return self.criterion

    def set_params(self, criterion='LS'):
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
    x = core.Matrix([[1, 2], [2, 2], [3, 1], [4, 2], [6, 6]])
    y = core.Vector([4, 6, 6.5, 9, 19])
    model = LinearRegression()
    model.fit(x, y)
    # print(model.coefficients)
    # print(model.predict(core.Matrix([[2, 3], [3, 3]])))
    # print(model.score(x, y))
    # print(reg_eval.correlation(x.col(0), y))
    model.evaluate(x, y)
