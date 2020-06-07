import linear_algebra.linear_algebra_core as core

def format_score(score, number_type=None):
    if number_type is None:
        return score
    else:
        return number_type(score)


def total_sum_of_squares(y_true, number_type=float, **kwargs):
    score = ((y_true - y_true.mean()) ** 2).sum()
    return format_score(score, number_type)


def residual_sum_of_squares(y_true, y_pred, number_type=float, **kwargs):
    score = ((y_true - y_pred) ** 2).sum()
    return format_score(score, number_type)


def residual_standard_error(y_true, y_pred, number_type=float, **kwargs):
    score = ((1 / (len(y_true) - 2)) *
             residual_sum_of_squares(y_true, y_pred, number_type)) ** 0.5
    return format_score(score, number_type)


def correlation(x, y, number_type=float, **kwargs):
    """
    
    :param x: 
    :param y: 
    :param number_type: 
    :param kwargs: 
    :return: 
    """
    score = ((x - x.mean()).transpose() * (y - y.mean())) / \
            (total_sum_of_squares(x, number_type) ** 0.5 *
             total_sum_of_squares(y, number_type) ** 0.5)
    return format_score(score, number_type)


def f_statistic(y_true, y_pred, num_predictors, number_type=float, **kwargs):
    if (len(y_true) - num_predictors - 1) <= 0:
        raise ValueError('Not enough y values to determine F-statistic')
    score = (total_sum_of_squares(y_true, number_type) -
             residual_sum_of_squares(y_true, y_pred, number_type)) / \
            num_predictors / \
            residual_sum_of_squares(y_true, y_pred, number_type) / \
            (len(y_true) - num_predictors - 1)
    return format_score(score, number_type)


def chi_squared(y_true, y_pred, number_type=float, **kwargs):
    score = ((y_pred - y_true.mean())**2) / y_true.mean()
    return format_score(score, number_type)


def standard_error_coefs(x, y_true, y_pred, number_type=float, **kwargs):
    x = core.enhance_matrix(x)
    matrix = (x.transpose()*x).inverse()
    s2 = (y_true - y_pred).transpose() * (y_true - y_pred) / \
         (len(y_true) - x.n_cols)
    return [(s2 * matrix.data[i][i])**2 for i in range(matrix.n_cols)]


def t_statistic(x, y_true, y_pred, coefs, number_type=float, **kwargs):
    x = core.enhance_matrix(x)
    se_coefs = standard_error_coefs(x, y_true, y_pred, number_type)
    return [coefs[i]/se_coefs[i] for i in range(len(coefs))]
