import modeling.evaluation.regression_evaluation as reg_eval


def mean_squared_error(y_true, y_pred, number_type=float):
    """
    Mean squared error of predictions
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = ((y_true - y_pred) ** 2).sum() / len(y_true)
    return reg_eval.format_score(score, number_type)


def mean_absolute_error(y_true, y_pred, number_type=float):
    """
    Mean absolute error of predictions
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = (y_true - y_pred).abs().sum() / len(y_true)
    return reg_eval.format_score(score, number_type)


def r_squared(y_true, y_pred, number_type=float):
    """
    R^2 score of prediction
    :param y_true: Vector
    :param y_pred: Vector
    :param number_type: type of output function
    :return: number
    """
    score = 1 - reg_eval.residual_sum_of_squares(y_true, y_pred, number_type) / \
            reg_eval.total_sum_of_squares(y_true, number_type)
    return reg_eval.format_score(score, number_type)


def reduced_chi_squared(y_true, y_pred, num_predictors, number_type=float):
    score = reg_eval.chi_squared(y_true, y_pred, number_type) / \
            (len(y_true) - num_predictors)
    return reg_eval.format_score(score, number_type)
