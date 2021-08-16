import abc
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
from itca.utils import bidict, compute_y_dist
from itca.metrics import itca


class Node(object):
    def __init__(self, vals, mapping):
        self.vals = vals
        self.mapping = mapping
        self.children = None

    def __repr__(self):
        return "{}|ITAC={:.4f}|".format(self.mapping, np.mean(self.vals))

    def add_child(self, node):
        if self.children:
            self.children.append(node)
        else:
            self.children = [node]


class Strategy(metaclass=abc.ABCMeta):
    """
    Attributes
    ---------
    class_type: str
        The class tyype. It should be one of "nominal" (the class has no specific oder), "ordinal" and "tree"
        for tree structured (not implemented for now).

    metric: function
        The function with signature (y_test, y_pred, mapping, y_dist) and outputs a real-value.


    Methods
    -------
    compute_itca(self, X, y, clf, mapping, kfolds=5, y_dist=None, return_class_acc=False)
        Given the map compute the metric value.

    search(self, X, y, clf, kfolds=5, verbose=False, early_stop=True):
        Search the best mapping start from the identitfy map.
    """
    def __init__(self, class_type="nominal", metric=itca):
        """
        Searching strategy.

        Parameters
        ----------
        class_type: {"nominal", "ordinal", "tree"}
            Type of classes.
        """
        self.class_type = class_type
        self.metric = metric
        self.path = None
        self.selected = None

    def __repr__(self):
        return "{}(class_type={})".format(self.__class__.__name__, self.class_type)

    @staticmethod
    def combine_i_j(i, j, mapping):
        """
        Combine (merged) class labels i and j in mapping.

        Parameters
        ----------
        i: int
        j: int

        Return
        ------
        next_mapping: bidict
            mapping that combines the class i and j
        
        Example
        -------
        >>mapping = bidict({0:0, 1:1, 2:2})
        >>self.combine_i_j(0, 1, mapping)
        {0:0, 1:0, 2:1}
        """
        next_mapping = dict()
        for key in mapping.inverse:
            if key < j:
                next_mapping.update({e: key for e in mapping.inverse[key]})
            elif key > j:
                next_mapping.update({e: key - 1 for e in mapping.inverse[key]})
            else:
                next_mapping.update({e: i for e in mapping.inverse[key]})
        return bidict(next_mapping)

    def next_mappings(self, mapping, class_type):
        """
        Generate mappings considered in the next round of search.

        Parameters
        ----------
        mapping: bidict
            classes combination mappings.

        class_type: {"nominal", "ordinal", "tree"}
            Type of classes.

        Yields
        ------
        bidict
            Mappings that are considered in the next round of search.

        Raises
        ------
        ValueError
            If class_type is illegal.
        """
        if class_type == "nominal":
            for i, j in combinations(range(len(mapping.inverse)), 2):
                yield self.combine_i_j(i, j, mapping)
        elif class_type == "ordinal":
            for i in range(len(mapping.inverse) - 1):
                yield self.combine_i_j(i, i + 1, mapping)
        elif class_type == "tree":
            raise NotImplementedError(
                "Tree-based next-mapping is not implemeted")
        else:
            raise ValueError(
                "class_type must be one of \"nominal\", \"ordinal\" and \"tree\".")

    def next_mappings_pruned(self, mapping, class_type, class_acc, ty_dist):
        """
        Generate mappings considered in the next round of search.

        Parameters
        ----------
        mapping: bidict
            classes combination mappings.

        class_type: {"nominal", "ordinal", "tree"}
            Type of classes.

        Yields
        ------
        bidict
            Mappings that are considered in the next round of search.

        Raises
        ------
        ValueError
            If class_type is illegal.
        """

        def compute_lb(i, j):
            ai, aj = np.mean(class_acc[i]), np.mean(class_acc[j])
            pi, pj = ty_dist[i], ty_dist[j]
            return (-pi * np.log(pi) * np.mean(ai) - pj * np.log(pj) * np.mean(aj)) \
                   / ((-pi - pj) * np.log(pi + pj))

        if class_type == "nominal":
            for i, j in combinations(range(len(mapping.inverse)), 2):
                if compute_lb(i, j) < 1:
                    yield self.combine_i_j(i, j, mapping)
        elif class_type == "ordinal":
            for i in range(len(mapping.inverse) - 1):
                if compute_lb(i, i + 1) < 1:
                    yield self.combine_i_j(i, i + 1, mapping)
        elif class_type == "tree":
            raise NotImplementedError(
                "Tree-based next-mapping is not implemeted")
        else:
            raise ValueError(
                "class_type must be one of \"nominal\", \"ordinal\" and \"tree\".")

    def compute_itca(self, X, y, clf, mapping, kfolds=5, y_dist=None, return_class_acc=False):
        """
        Given the data and a classifier, compute ITCA through cross validation.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.

        y: array like of shape (n_samples, )
            The target class labels. Integers from 0 to K-1.

        clf: scikit-learn like classifier 
            Classifiers that implements the standard interfaces of `scikit-learn`, 
            including `fit(X, y)` and `predict(X)`.

        kfolds: int or sklearn.model_selection.KFold instance
            Folds of cross validation.

        y_dist: dict, optional
            The classes distribution of y. If y_dist is None, this method will 
            compute it automatically.

        return_class_acc: bool
            Return the class-wise accuracy.

        Returns
        -------
        itca_cv: (kfolds, ) ndarray
            The cross validated ITCA.ndarray of ITCA.
        
        class_acc: dict of accuracy of each class.
            Accruacy of each class.

        Raises
        ------
        ValueError
            kf is illegal.
        
        ValueError
            y_dist is illegal.
        """
        if isinstance(kfolds, int) and kfolds > 1:
            kf = KFold(n_splits=kfolds, shuffle=True)
        elif kfolds == 1:
            pass
        elif isinstance(kfolds, KFold):
            kf = kfolds
        else:
            raise ValueError("kf must be int or sklearn.model_selection.KFold instance.")
        if y_dist is None:
            y_dist = compute_y_dist(y)
        itca_cv = []
        ty = mapping.map(y)
        unique_ty = np.unique(ty)
        if return_class_acc:
            class_acc = {key: [] for key in unique_ty}
        if kfolds == 1:
            clf.fit(X, ty)
            y_pred = clf.predict(X)
            itca_cv = [self.metric(y, y_pred, mapping, y_dist)]
        else:
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                ty_train, y_test = ty[train_index], y[test_index]
                clf.fit(X_train, ty_train)
                y_pred = clf.predict(X_test)
                itca_cv.append(self.metric(y_test, y_pred, mapping, y_dist))
                ty_test = ty[test_index]
                if return_class_acc:
                    for key in class_acc:
                        indexes = ty_test == key
                        class_acc[key].append(np.sum(ty_test[indexes] == y_pred[indexes]) / np.sum(indexes))

        if return_class_acc:
            return np.array(itca_cv), class_acc
        else:
            return np.array(itca_cv)

    @abc.abstractmethod
    def search(self, X, y, clf, kfolds=5, verbose=False, early_stop=False):
        return self


class GreedySearch(Strategy):
    """
    Search the class combination map that maximizes s-ITCA by greedy algorithm.
    """

    def search(self, X, y, clf, kfolds=5, verbose=False, early_stop=True):
        """
        Search the best mapping start.

        Parameters
        ----------
        X: array like of shape (n_samples, n_features)
            The training input samples.

        y: array like of shape (n_samples, )
            The target class labels. Integers from 0 to K-1.

        clf: scikit-learn like classifier
            Classifiers that implements the standard interfaces of `scikit-learn`,
            including `fit(X, y)` and `predict(X)`.

        early_stop: bool
            Stop when the metric cannot be improved by class combination when `early_stop` is True.

        Return
        ------
        self: object
        """
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        if min(y) != 0 or max(y) != n_classes - 1:
            raise ValueError("Labels should be between 0 to K.")
        if kfolds > 1:
            kf = KFold(n_splits=kfolds, shuffle=True)
        else:
            kf = 1
        # compute y_dist
        y_dist = compute_y_dist(y)
        # construct identity mapping.
        cur_mapping = bidict({key: key for key in range(n_classes)})
        cur_itca = self.compute_itca(X, y, clf, cur_mapping, kfolds=kf, y_dist=y_dist)
        cur_node = Node(cur_itca, cur_mapping)
        self.selected = cur_node
        self.path = cur_node
        i = 1
        self.counter = 0
        while len(cur_mapping.inverse) > 2:
            # compute itac of current mapping
            if verbose:
                print("Round={}|Current Mapping={}".format(i, cur_mapping))
            next_itcas = []
            candidate_mappings = []
            for next_mapping in self.next_mappings(cur_mapping, self.class_type):
                candidate_mappings.append(next_mapping)
                self.counter += 1
                next_itcas.append(self.compute_itca(X, y, clf, next_mapping, kfolds=kf, y_dist=y_dist))
            # choose best mapping
            next_ind = np.argmax([np.mean(itca) for itca in next_itcas])
            cur_mapping = candidate_mappings[next_ind]
            next_node = Node(next_itcas[next_ind], cur_mapping)
            cur_node.children = [next_node]
            cur_node = next_node
            i += 1
            if np.mean(cur_node.vals) > np.mean(self.selected.vals):
                self.selected = cur_node
            elif early_stop:
                break
        return self


class GreedySearchPruned(Strategy):
    """
    Search the class combination map that maximizes s-ITCA by greedy algorithm.
    Using the lower bound derived by class-wise accuracy to prune the search space.
    """

    def search(self, X, y, clf, kfolds=5, verbose=False, early_stop=True):
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        if min(y) != 0 or max(y) != n_classes - 1:
            raise ValueError("Labels should be between 0 to K.")
        kf = KFold(n_splits=kfolds, shuffle=True)
        # compute y_dist
        y_dist = compute_y_dist(y)
        # construct identity mapping.
        cur_mapping = bidict({key: key for key in range(n_classes)})
        cur_itca, cur_class_acc = self.compute_itca(X, y, clf, cur_mapping, kfolds=kf, y_dist=y_dist,
                                                    return_class_acc=True)
        cur_y_dist = compute_y_dist(cur_mapping.map(y))
        cur_node = Node(cur_itca, cur_mapping)
        self.selected = cur_node
        self.path = cur_node
        i = 1
        self.counter = 0
        while len(cur_mapping.inverse) > 2:
            # compute itac of current mapping
            if verbose:
                print("Round={}|Current Mapping={}".format(i, cur_mapping))
            next_itcas = []
            next_class_accs = []
            candidate_mappings = []
            for next_mapping in self.next_mappings_pruned(cur_mapping, self.class_type,
                                                          cur_class_acc, cur_y_dist):
                candidate_mappings.append(next_mapping)
                self.counter += 1
                itca, class_acc = self.compute_itca(X, y, clf, next_mapping, kfolds=kf, y_dist=y_dist,
                                                    return_class_acc=True)
                next_itcas.append(itca)
                next_class_accs.append(class_acc)
            # choose best mapping
            if len(next_itcas) == 0:
                break
            else:
                next_ind = np.argmax([np.mean(itca) for itca in next_itcas])
                cur_mapping = candidate_mappings[next_ind]
                cur_class_acc = next_class_accs[next_ind]
                cur_y_dist = compute_y_dist(cur_mapping.map(y))
                next_node = Node(next_itcas[next_ind], cur_mapping)
                cur_node.children = [next_node]
                cur_node = next_node
                i += 1
                if np.mean(cur_node.vals) > np.mean(self.selected.vals):
                    self.selected = cur_node
        return self


class BFSearch(Strategy):
    """
    Search the class combination map that maximizes s-ITCA by breadth-first search algorithm.
    """
    def search(self, X, y, clf, kfolds=5, verbose=False, early_stop=False):
        self.counter = 0

        def bfs(visited, cur_node, kf, y_dist, i):
            if len(cur_node.mapping.inverse) > 2:
                for next_mapping in self.next_mappings(cur_node.mapping, self.class_type):
                    next_fs = frozenset(next_mapping.items())
                    if next_fs not in visited:
                        self.counter += 1
                        next_itca = self.compute_itca(X, y, clf, next_mapping, kf, y_dist)
                        visited.add(next_fs)
                        if np.mean(next_itca) > np.mean(cur_node.vals):
                            next_node = Node(next_itca, next_mapping)
                            cur_node.add_child(next_node)
                            if verbose:
                                print("Level={}|Current Mapping={}".format(i, next_mapping))
                            if np.mean(next_itca) > np.mean(self.selected.vals):
                                self.selected = next_node
                            bfs(visited, next_node, kf, y_dist, i + 1)

        unique_y = np.unique(y)
        n_classes = len(unique_y)
        if min(y) != 0 or max(y) != n_classes - 1:
            raise ValueError("Labels should be between 0 to K.")
        kf = KFold(n_splits=kfolds, shuffle=True)
        # compute y_dist
        y_dist = compute_y_dist(y)
        # construct identity mapping.
        cur_mapping = bidict({key: key for key in range(n_classes)})
        cur_itca = self.compute_itca(X, y, clf, cur_mapping, kfolds=kf, y_dist=y_dist)
        self.counter += 1
        cur_node = Node(cur_itca, cur_mapping)
        self.selected = cur_node
        self.path = cur_node
        i = 1
        visited = set([frozenset(cur_mapping.items())])
        # driver code
        bfs(visited, cur_node, kf, y_dist, i)
        self.visited = visited
        return self


class BFSearchPruned(Strategy):
    """
    Search the class combination map that maximizes s-ITCA by breadth-first search algorithm.
    Using the lower bound derived by class-wise accuracy to prune the search space.
    """
    def search(self, X, y, clf, kfolds=5, verbose=False, early_stop=True):
        self.counter = 0
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        if min(y) != 0 or max(y) != n_classes - 1:
            raise ValueError("Labels should be between 0 to K.")
        kf = KFold(n_splits=kfolds, shuffle=True)
        # compute y_dist
        y_dist = compute_y_dist(y)

        def bfs(visited, cur_node, cur_class_acc, cur_ty_dist, i):
            if len(cur_node.mapping.inverse) > 2:
                for next_mapping in self.next_mappings_pruned(cur_node.mapping, self.class_type,
                                                              cur_class_acc, cur_ty_dist):
                    next_fs = frozenset(next_mapping.items())
                    if next_fs not in visited:
                        self.counter += 1
                        next_itca, next_class_acc = self.compute_itca(X, y, clf, next_mapping, kf,
                                                                      y_dist, return_class_acc=True)
                        visited.add(next_fs)
                        if np.mean(next_itca) > np.mean(cur_node.vals):
                            next_node = Node(next_itca, next_mapping)
                            cur_node.add_child(next_node)
                            if verbose:
                                print("Level={}|Current Mapping={}".format(i, next_mapping))
                            if np.mean(next_itca) > np.mean(self.selected.vals):
                                self.selected = next_node
                            cur_ty_dist = compute_y_dist(next_node.mapping.map(y))
                            bfs(visited, next_node, next_class_acc, cur_ty_dist, i + 1)

        # construct identity mapping.
        cur_mapping = bidict({key: key for key in range(n_classes)})
        cur_itca, cur_class_acc = self.compute_itca(X, y, clf, cur_mapping, kfolds=kf,
                                                    y_dist=y_dist, return_class_acc=True)
        self.counter += 1
        cur_node = Node(cur_itca, cur_mapping)
        self.selected = cur_node
        self.path = cur_node
        i = 1
        visited = set([frozenset(cur_mapping.items())])
        bfs(visited, cur_node, cur_class_acc, y_dist, i)
        self.visited = visited
        return self

if __name__ == "__main__":
    mapping = bidict({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
    for m in Strategy.next_mappings(mapping, "nominal"):
        print(m.inverse)
