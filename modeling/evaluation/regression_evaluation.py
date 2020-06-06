def format_score(score, number_type=None):
    if number_type is None:
        return score
    else:
        return number_type(score)


def total_sum_of_squares(y_true, number_type=float):
    score = ((y_true - y_true.mean()) ** 2).sum()
    return format_score(score, number_type)


def residual_sum_of_squares(y_true, y_pred, number_type=float):
    score = ((y_true - y_pred) ** 2).sum()
    return format_score(score, number_type)


def residual_standard_error(y_true, y_pred, number_type=float):
    score = ((1 / (len(y_true) - 2)) *
             residual_sum_of_squares(y_true, y_pred, number_type)) ** 0.5
    return format_score(score, number_type)


def correlation(x, y, number_type=float):
    print(x, y)
    score = ((x - x.mean()).transpose() * (y - y.mean())) / \
            (total_sum_of_squares(x, number_type) ** 0.5 *
             total_sum_of_squares(y, number_type) ** 0.5)
    return format_score(score, number_type)


def f_statistic(y_true, y_pred, num_predictors, number_type=float):
    score = (total_sum_of_squares(y_true, number_type) -
             residual_sum_of_squares(y_true, y_pred, number_type)) / \
            num_predictors / \
            residual_sum_of_squares(y_true, y_pred, number_type) / \
            (len(y_true) - num_predictors - 1)
    return format_score(score, number_type)


def chi_squared(y_true, y_pred, number_type=float):
    score = ((y_pred - y_true.mean())**2) / y_true.mean()
    return format_score(score, number_type)


def standard_error_coefs(x, y_true, y_pred, number_type=float):
    matrix = (x.transpose()*x).inverse()
    s2 = (y_true - y_pred).transpose() * (y_true - y_pred) / \
         (len(y_true) - x.n_cols)
    return [(s2 * matrix.data[i, i])**2 for i in range(matrix.n_cols)]
