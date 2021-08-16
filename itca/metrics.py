import numpy as np
from itca.utils import compute_y_dist

EPS = 1e-15


def _check_consistency(y_ori, y_mer, transform):
    n_classes_ori = max(transform.keys()) + 1
    n_classes_mer = max(transform.values()) + 1
    assert (n_classes_mer <= n_classes_ori)
    assert (len(transform) == n_classes_ori)
    return n_classes_ori, n_classes_mer


def class_accuracy(y_true, y_pred):
    ca = dict()
    unique_y_true = np.unique(y_true)
    for yi in unique_y_true:
        indices = y_true == yi
        ca[yi] = np.sum(y_true[indices] == y_pred[indices]) / np.sum(indices)
    return ca


def adjusted_accuracy_score(y_true, y_pred, mapping, adjust_by="cardinal"):
    """
    Adjusted accuracy is an adjustment for accuracy. 

    Parameters
    ----------
    y_true: int array, shape = [n_samples, ]
        A clustering of the data into disjoint subsets.

    y_pred: int array, shape = [n_samples, ]
        A clustering of the data into disjoint subsets.

    mapping: `bidict`, size = n_classes_ori
        transform the original labels to the merged labels.
    
    adjust_by: str, "cardinal" or "size"

    Returns
    -------
    aac: float ( 0 <= acc <= 1)
    """
    aac = 0
    _, n_classes_mer = _check_consistency(y_true, y_pred, mapping)
    y_t = np.array([mapping[yi] for yi in y_true])
    n_samples = y_true.size
    if adjust_by == "cardinal":
        weights = []
        for i in range(n_classes_mer):
            weights.append(len(mapping.inverse[i]))
        weights = 1.0 / np.array(weights)
    elif adjust_by == "size":
        y_t_dist = compute_y_dist(y_t)
        weights = 1.0 / np.array([y_t_dist[ind] for ind in y_t_dist])
        weights =  weights
    else:
        raise ValueError("Unknown ajust_by value. Adjust by cardinal or size!")
    aac = np.sum((y_t == y_pred) * weights[y_t]) / n_samples
    return aac



def itca(y_true, y_pred, mapping, y_dist=None):
    """
    Sample-level information-theoretic classification accuracy (s-ITCA)

    Parameters
    ----------
    y_true: int array, shape = [n_samples, ]
        A clustering of the data into disjoint subsets.

    y_pred: int array, shape = [n_samples, ]
        A clustering of the data into disjoint subsets.

    mapping: `bidict`, size = n_classes_ori
        Transform the original labels to the merged labels.

    y_dist: dict of length n_classes
        Distribution of the original classes.

    Returns
    -------
    itca_score: float
    """
    n_classes_ = len(mapping.inverse)
    if y_dist:
        if len(y_dist) != len(mapping):
            raise ValueError(
                "The len of y_dist and mapping must be the same.")
        y_dist_sum = 0
        for key in y_dist:
            pi = y_dist[key]
            assert (pi > 0)
            y_dist_sum += pi
        if abs(y_dist_sum - 1.0) > 1e-10:
            raise ValueError("The values of y_dist must be summing to 1.")
    else:
        y_dist = compute_y_dist(y_true)
    y_dist_tf = dict()
    for key in mapping.inverse:
        y_dist_tf[key] = 0
        ori_classes = mapping.inverse[key]
        for ori_class in ori_classes:
            y_dist_tf[key] += y_dist[ori_class]
    ty = mapping.map(y_true)
    weights = np.array([-np.log(y_dist_tf[i]) for i in range(n_classes_)])
    den = weights[ty]
    num = den[ty == y_pred]
    return np.sum(num) / den.size


def prediction_entropy(y_true, y_pred, mapping, y_dist=None):
    """
    Compute the entropy of  H(mapping(y_true), y_pred).

    Parameters
    ----------
    y_true: numpy array, (n_samples, )
        True labels.

    y_pred: numpy array, (n_samples, )
        Prediction labels.

    mapping: bidict of size n_classes
        Class mapping.

    Returns
    -------
    float
        The value of the prediction entropy
    """
    n_classes_ = len(mapping.inverse)
    if y_dist:
        if len(y_dist) != len(mapping):
            raise ValueError(
                "The len of y_dist and transform must be the same.")
        y_dist_sum = 0
        for key in y_dist:
            pi = y_dist[key]
            assert (pi > 0)
            y_dist_sum += pi
        if abs(y_dist_sum - 1.0) > 1e-10:
            raise ValueError("The values of y_dist must be summing to 1.")
    else:
        y_dist = compute_y_dist(y_true)
    ty = mapping.map(y_true)
    ca = class_accuracy(ty, y_pred)
    h = 0
    for key in ca:
        if ca[key] > EPS:
            p = np.sum([y_dist[i] for i in mapping.inverse[key]])
            h += - np.log(p * ca[key]) * p * ca[key]
    return h


if __name__ == "__main__":
    from itca import bidict
    m = bidict({0: 0, 1: 0, 2: 1})
    y_true = np.random.randint(0, 3, 1000)
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.random.randint(0, 2, 1000)
    y_pred = m.map(y_true)
    y_dist = {0: .3, 1: .4, 2: .3}
    acc1 = adjusted_accuracy_score(y_true, y_pred, m, adjust_by="cardinal")
    acc2 = adjusted_accuracy_score(y_true, y_pred, m, adjust_by="size")

