import itertools
import numpy as np


def compute_y_dist(y):
    """
    Compute the distribution of labels

    Parameters
    ----------
    y: numpy array
        Class labels  vector whose values range from 0 to K - 1

    Returns
    -------
    dict
        A dictionary whose keys indicate the class labels and values indicate the corresponding proportions.

    """
    keys, vals = np.unique(y, return_counts=True)
    vals = vals.astype(float) / y.size
    y_dist = {keys[i]: vals[i] for i in range(keys.size)}
    return y_dist


class bidict(dict):
    """
    Bidirectional dictionary inherited from the built-in `dict` to represent the class combination mapping.
    """

    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __eq__(self, b):
        if not isinstance(b, bidict):
            raise TypeError("__eq__ method requires both objects to be bidict")
        return frozenset(self.items()) == frozenset(b.items())

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

    def map(self, arr):
        return np.array([self.__getitem__(xi) for xi in arr])

    def reverse_map(self, arr):
        n_classes_ori = np.unique(arr).size
        n_classes_ext = len(self)
        if n_classes_ext < n_classes_ori:
            raise ValueError("The number of extended glasses shoulb be greater \
                than the number of the original classes")
        if isinstance(arr, list):
            arr_size = len(arr)
        elif isinstance(arr, np.ndarray):
            arr_size = arr.size
        else:
            TypeError("arr should be ndarray or list of integers.")
        y_ext = np.zeros(arr_size)
        for ori_label in self.inverse:
            ind_labels = arr == ori_label
            y_ext[ind_labels] = np.random.choice(self.inverse[ori_label], size=np.sum(ind_labels))
        return y_ext.astype(int)


def bv2transformation(v):
    """
    Convert binary vector to transformation.
    """
    n_bars = v.size
    n_classes = n_bars + 1
    tf = bidict()
    j = 0
    for i in range(n_classes):
        tf[i] = j
        if i < n_bars and v[i] == 1:
            j += 1
    return tf


def int2bvstr(classes_num, a):
    if 2 ** classes_num - 1 < a:
        raise ValueError("a exceeds the 2**classes_num - 1")
    return "{1:0{0}b}".format(classes_num - 1, a)


def bvstr2bv(bvstr):
    return np.array([int(i) for i in bvstr])


def int2mapping(observed_classes, i):
    true_bvstr = int2bvstr(observed_classes, i)
    true_bv = bvstr2bv(true_bvstr)
    true_mapping = bv2transformation(true_bv)
    return true_mapping


def compute_hamming_distance(v1, v2):
    """
    Compute the Hamming distance between two binary strings or binary vector

    Parameters
    ----------
    v1: str or binary numpy array
    v2: str or binary numpy array

    Returns
    -------
    int
        The Hamming distance between the two vectors.
    """
    if isinstance(v1, str) and isinstance(v2, str):
        bv1 = bvstr2bv(v1)
        bv2 = bvstr2bv(v2)

    elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        bv1 = v1
        bv2 = v2
    else:
        raise TypeError("Arguments should be string or numpy.array")
    return np.sum(np.logical_not(bv1 == bv2))


def enumerate_transforms(n_classes_ori, n_classes_mer):
    """
    Enumerate all 
    """
    n_bars = n_classes_ori - n_classes_mer
    if n_bars <= 0:
        raise ValueError("n_classes_ori should be greater than n_classes_mer!")
    total_bars = n_classes_ori - 1
    for bars in itertools.combinations(list(range(total_bars)), n_bars):
        bars = np.array(bars)
        transform = dict()
        for cur_classes_ext in range(n_classes_ori):
            n_left_bars = np.sum(cur_classes_ext > bars).astype(int)
            transform.update({cur_classes_ext: cur_classes_ext - n_left_bars})
        yield bidict(transform)


def prob_support(labels):
    """
    Convert array-like labels to (n_samples, n_classes) probility support.
    """
    n_classes = np.unique(labels).size
    n_samples = labels.size
    pred_sup = np.zeros([n_samples, n_classes])
    pred_sup[list(range(n_samples)), labels] = 1.0
    return pred_sup


def perm_labels(y_true, n_samples, accuracy):
    ind = np.random.choice(n_samples, size=np.ceil((1 - accuracy) * n_samples).astype(int), replace=False)
    labels = np.unique(y_true)
    y_perm = y_true.copy()
    for label in labels:
        ind_label = ind[y_true[ind] == label]
        y_perm[ind_label] = np.random.choice(np.delete(labels, label), size=ind_label.size)
    return y_perm


def inv_logit(x):
    """
    Inverse logit function.
    """
    return np.exp(x) / (1 + np.exp(x))


def extend_classes(y_ori, transform):
    """
    Extend the original labels y_ori to the extended labels by transform.
    """
    n_classes_ori = np.unique(y_ori).size
    n_classes_ext = len(transform)
    if n_classes_ext <= n_classes_ori:
        raise ValueError("The number of extended glasses shoulb be greater \
            than the number of the original classes")
    y_ext = np.zeros(y_ori.size)
    for ori_label in transform.inverse:
        ind_labels = y_ori == ori_label
        y_ext[ind_labels] = np.random.choice(transform.inverse[ori_label], size=np.sum(ind_labels))
    return y_ext.astype(int)


if __name__ == "__main__":
    bvstr1 = int2bvstr(5, 4)
    bvstr2 = int2bvstr(5, 14)
    bv1 = bvstr2bv(bvstr1)
    bv2 = bvstr2bv(bvstr2)
    print(compute_hamming_distance(bv1, bv2))
