import modeling.helpers.regression_evaluation as reg_eval


def mean_squared_error(y_true, y_pred, number_type=float, **kwargs):
    """
    Mean squared error of predictions
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = ((y_true - y_pred) ** 2).sum() / len(y_true)
    return reg_eval.format_score(score, number_type)


def root_mean_squared_error(y_true, y_pred, number_type=float, **kwargs):
    """
    Root of mean squared error of predictions
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = mean_squared_error(y_true, y_pred, number_type) ** 0.5
    return reg_eval.format_score(score, number_type)


def mean_absolute_error(y_true, y_pred, number_type=float, **kwargs):
    """
    Mean absolute error of predictions
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = (y_true - y_pred).abs().sum() / len(y_true)
    return reg_eval.format_score(score, number_type)


def r_squared(y_true, y_pred, number_type=float, **kwargs):
    """
    R^2 score of prediction
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    rss = reg_eval.residual_sum_of_squares(y_true, y_pred, number_type)
    tss = reg_eval.total_sum_of_squares(y_true, number_type)
    score = 1 - rss / tss
    return reg_eval.format_score(score, number_type)


def reduced_chi_squared(y_true, y_pred, num_predictors, number_type=float,
                        **kwargs):
    score = reg_eval.chi_squared(y_true, y_pred, number_type) / \
            (len(y_true) - num_predictors)
    return reg_eval.format_score(score, number_type)
