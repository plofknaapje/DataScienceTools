import linear_algebra.linear_algebra_core as core
import modeling.helpers.regression_metrics as reg_met
import modeling.helpers.regression_evaluation as reg_eval
import modeling.helpers.nn_distances as nn_dist
from tabulate import tabulate
from scipy.stats import f, t


# TODO: add preprocessing for categorical variables
# TODO: document all functions

class Model:

    def fit(self, x, y_true):
        raise Exception('Fitting not implemented')

    def predict(self, x):
        raise Exception('Predicting not implemented')

    def evaluate(self, x, y_true):
        raise Exception('Evaluation not implemented')

    def set_params(self, parameters):
        raise Exception('Not implemented')

    def get_params(self):
        raise Exception('Not implemented')


class LinearRegression(Model):

    def __init__(self, criterion='LS'):
        self.criterion = criterion
        self.coefficients = None

    def fit(self, x, y_true):
        """
        Fits the parameters of X to predict the value y with model criterion
        :param x: Matrix with equal amounts of rows as y
        :param y_true: Column Vector with length equal to rows in X
        """
        if self.criterion != 'LS':
            raise Exception('Only Least Squares is implemented')
        reg_eval.regression_data_check(x, y_true)
        x = core.enhance_matrix(x)

        self.coefficients = (x.transpose() * x).inverse() * x.transpose() * y_true

    def predict(self, x):
        if self.coefficients is None:
            raise Exception('Model has not been fitted yet')
        x = core.enhance_matrix(x)
        reg_eval.regression_data_check(x, width=len(self.coefficients))

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
                var_outcomes[i[0]] = [2 * t.sf(abs(float(score)), len(y_pred) - x.n_cols)
                                      for score in var_outcomes['t-statistic']]

        print(tabulate(glob_outcomes, headers='keys'))
        print(tabulate(var_outcomes, headers='keys'))

        return glob_outcomes, var_outcomes

    def get_params(self):
        return self.criterion

    def set_params(self, criterion='LS'):
        self.criterion = criterion


class KNNRegression(Model):

    def __init__(self, weight='equal', measure='euclidean'):
        self.weight = weight
        self.measure = measure
        self.k = None
        self.x = None
        self.y_true = None

    def fit(self, x, y_true, k=5):
        self.x = x
        self.y_true = y_true
        if k > x.n_rows:
            raise Exception('k is larger than the amount of data points in x')
        self.k = k

    def predict(self, x):
        distances = [[[nn_dist.distance_continuous(x.row(new_row),
                                                  self.x.row(train_row),
                                                  self.measure),
                      train_row] for train_row in range(self.x.n_rows)]
                     for new_row in range(x.n_rows)]
        distances = [sorted(row)[:self.k] for row in distances]
        
        if self.weight == 'equal':
            predictions = [sum(self.y_true.data[i[1]] for i in row)/self.k
                           for row in distances]
        
        return core.Vector(predictions)


def knn_error(x, y_true, k, weight='equal'):
    distances_matrix = x * x.transpose()
    distances_list_pairs = [sorted([[row[i], i] for i in range(len(row))])[0:k]
                            for row in distances_matrix.data]
    print(distances_list_pairs)
    y_pred_list = [sum([i[1] for i in row]) / k for row in distances_list_pairs]
    y_pred = core.Vector(y_pred_list)
    return reg_met.root_mean_squared_error(y_pred, y_true)


if __name__ == "__main__":
    x = core.Matrix([[1, 2], [2, 2], [3, 1], [4, 2], [6, 6]])
    y = core.Vector([4, 6, 6.5, 9, 19])
    x_test = core.Matrix([[1, 1], [2, 2], [3, 3]])
    y_test = core.Vector([3, 6.5, 9])
    # model = LinearRegression()
    # model.fit(x, y)
    # print(model.coefficients)
    # print(model.predict(core.Matrix([[2, 3], [3, 3]])))
    # print(model.score(x, y))
    # print(reg_eval.correlation(x.col(0), y))
    # model.evaluate(x, y)
    knn = KNNRegression()
    
    for k in range(1, 4):
        knn.fit(x, y, k=k)
        y_pred = knn.predict(x_test)
        print(reg_met.mean_absolute_error(y_test, y_pred))
    
