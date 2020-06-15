import linear_algebra.linear_algebra_core as core


def distance_continuous(vec_a, vec_b, method='euclidean'):
    if method == 'euclidean':
        return ((vec_a - vec_b) ** 2).sum() ** 0.5
    elif method == 'manhattan':
        return (vec_a - vec_b).abs().sum()
    else:
        raise Exception('Chosen method is not implemented')
