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


def f_statistic(num_predictors, y_true, y_pred, number_type=float):
    score = (total_sum_of_squares(y_true, number_type) - 
             residual_sum_of_squares(y_true, y_pred, number_type)) / \
            num_predictors / \
            residual_sum_of_squares(y_true, y_pred, number_type) / \
            (len(y_true) - num_predictors - 1)
    return format_score(score, number_type)

# def standard_error(x, y_true, y_pred, number_type=float):
#     var = residual_standard_error(y_true, y_pred, number_type)**2
#     se = []
#     for i in range(x.n_cols):
#         if i == 0:
#             se.append(var*(1/len(y_true) + ))
#         else:
#             se.append()
