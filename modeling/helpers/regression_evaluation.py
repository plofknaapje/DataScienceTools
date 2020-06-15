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


def residual_standard_error(y_true, y_pred, num_predictors, number_type=float, **kwargs):
    rss = residual_sum_of_squares(y_true, y_pred, number_type)
    score = ((1 / (len(y_true) - num_predictors)) * rss) ** 0.5
    return format_score(score, number_type)


def correlation(x, y, number_type=float, **kwargs):
    """
    :param x: 
    :param y: 
    :param number_type: 
    :param kwargs: 
    :return: 
    """
    tss = total_sum_of_squares(x, number_type)
    score = ((x - x.mean()).transpose() * (y - y.mean())) / \
            (tss**0.5 * tss**0.5)
    return format_score(score, number_type)


def f_statistic(y_true, y_pred, num_predictors, number_type=float, **kwargs):
    if (len(y_true) - num_predictors - 1) <= 0:
        raise ValueError('Not enough y values to determine F-statistic')
    tss = total_sum_of_squares(y_true, number_type)
    rss = residual_sum_of_squares(y_true, y_pred, number_type)
    df = len(y_true) - num_predictors
    score = ((tss - rss) / (num_predictors - 1)) / (rss / df)
    return format_score(score, number_type)


def chi_squared(y_true, y_pred, number_type=float, **kwargs):
    score = ((y_pred - y_true.mean())**2) / y_true.mean()
    return format_score(score, number_type)


def standard_error_coefs(x, y_true, y_pred, number_type=float, **kwargs):
    x = core.enhance_matrix(x)
    mse = ((y_true - y_pred)**2).sum() / (len(y_true) - x.n_cols)
    matrix = (x.transpose()*x).inverse()
    return [(mse * matrix.data[i][i])**0.5 for i in range(x.n_cols)]


def t_statistic(x, y_true, y_pred, coefs, number_type=float, **kwargs):
    x = core.enhance_matrix(x)
    se_coefs = standard_error_coefs(x, y_true, y_pred, number_type)
    return [coefs[i]/se_coefs[i] for i in range(len(coefs))]


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
