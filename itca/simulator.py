import numpy as np
from numpy import matlib, linalg
from itca.utils import bidict, perm_labels, extend_classes, inv_logit


class SimData(object):
    """
    Simulated data.
    """
    def __init__(self, generator, X, y, centroids=None, probs=None):
        """
        Simulated data. 

        Parameters
        ----------
        generator: `str`

        X: covariate matrix, shape = [n_samples, n_features]

        y: integer array that indicates labels, shape = [n_samples, ]

        probs: probability distribution matrix, shape = [n_classes, n_samples]
        """
        self.generator_name = generator
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.size = y.size
        assert(self.size == X.shape[0])
        self.n_classes = np.unique(y).size
        self.probs = probs
        self.centriods = centroids

    def __repr__(self):
        return "{}(n_features={}, n_samples={}, n_classes={})".format(self.generator_name, 
        self.n_features, self.n_samples, self.n_classes)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            if self.probs is not None:
                probs = self.probs[:, key]
            else:
                probs = None
            return SimData(self.generator_name, self.X[:, key], self.y[key], 
            probs=probs)
        elif isinstance(key, int):
            if key < 0 or key >= self.n_samples:
                raise IndexError("The index ({}) is out of range.".format(key))
            if self.probs is not None:
                probs = self.probs[:, key, np.newaxis]
            else:
                probs = None
            return SimData(self.generator_name, self.X[:, key, np.newaxis], 
            self.y[key], probs=probs)


    def concat(self, sim_data):
        assert(sim_data.generator_name == self.generator_name)
        if not isinstance(sim_data, SimData):
            raise ValueError("They type of parameter should be SimData.")
        y = np.concatenate([self.y, sim_data.y])
        X = np.concatenate([self.X, sim_data.X], axis=1)
        if self.probs is None or sim_data.probs is None:
            probs = None
        else:
            probs = np.concatenate([self.probs, sim_data.probs], axis=1)
        return SimData(self.generator_name, X, y, probs=probs)
    
    def shuffle(self):
        """
        Shuffle the simulated data
        """
        ind = np.arange(self.n_samples)
        np.random.shuffle(ind)
        self.X = self.X[ind, :]
        self.y = self.y[ind]
        if self.probs is not None:
            self.probs = self.probs[:, ind]
        return self



def sim_y(n_samples, n_classes, accuracy=0.9, seed=None, p=None):
    """
    Generate ground truth of labels and the prediction labels
    """
    if seed:
        np.random.seed(seed)
    y_true = np.random.choice(n_classes, n_samples, p=p)
    y_pred = perm_labels(y_true, n_samples, accuracy)
    return y_true, y_pred


def sim_merged_y(y_ori, transform, accuracy=0.9, seed=None, p=None):
    """
    Generate the merged clustering given the original labels `y_ori`, transform and accuracy. 
    """
    if seed:
        np.random.seed(seed)
    n_samples = y_ori.size
    y_mer = np.array([transform[yi] for yi in y_ori])
    y_mer_pred = perm_labels(y_mer, n_samples, accuracy)
    return y_mer, y_mer_pred

class SimOrdLogit(object):
    """
    Generate simulation objects from ordinal logistic regression model.
    """

    def __init__(self, n_classes, coeff, intercepts):
        """
        Initialize.
        
        Paramters
        ---------
        n_classes: `int`, number of classes.

        coeff: cofficients, shape = [n_features, 1]
        intercepts: intercepts, shape = [n_classes - 1, 1]
        """
        if n_classes < 2:
            raise ValueError("The number of classes is smaller than 2!")
        self.n_classes = n_classes
        self.labels = np.arange(n_classes)
        self.coeff = coeff[:, np.newaxis]
        self.n_features = coeff.shape[0]
        if intercepts.shape[0] != self.n_classes - 1:
            raise ValueError("intercepts is of size [n_classes -1, 1]")
        self.intercepts = intercepts[:, np.newaxis]
    
    def logodds_to_probs(self, logodds):
        """
        Compute probablilities by logodds.
        """
        _, n_samples = logodds.shape
        probs = np.zeros([self.n_classes, n_samples])
        inv_logodds = inv_logit(logodds)
        probs[-1, :] = inv_logodds[-1, :]
        if self.n_classes > 1:
            prob_pre = 1.0
            for ind_classes in range(self.n_classes - 1):
                probs[ind_classes, :] = prob_pre - inv_logodds[ind_classes, :]
                prob_pre = inv_logodds[ind_classes, :]
        return probs
    
    def probs_to_logodds(self, probs):
        n_classes, n_samples = probs.shape
        logodds = np.zeros([n_classes - 1, n_samples])
        for ind in range(n_classes - 1):
            logodds[ind, :] = np.sum(probs[ind+1:, :], 0) / np.sum(probs[:ind+1, :], 0)
        return np.log(logodds)

    def probs_to_x(self, probs):
        """
        Compute the covariates `x` by probability.

        Parameters
        ----------
        probs: shape = [n_classes, n_samples]
        """
        A = np.dot(self.coeff, self.coeff.T)
        logodds = self.probs_to_logodds(probs)
        x = np.dot(linalg.pinv(A), self.coeff * np.sum(logodds - self.intercepts)) \
              / (self.n_classes - 1.0)
        return x
    
    def x_to_logodds(self, x):
        """
        Compute log odds from covariates.
        """
        logodds = np.dot(self.coeff.T, x)
        logodds = matlib.repmat(logodds, self.n_classes - 1, 1) + self.intercepts
        return logodds

    def gen(self, n_samples, X=None, seed=None):
        """
        Generate data.

        Returns
        -------
        SimData(probs, y, X)
            probs: probability distribution, shape = [n_classes, n_samples]
            y: class labels, integer array
            X: covairates, shape = [n_features, n_samples]
        """
        if seed is not None: np.random.seed(seed)
        if X is None:
            X = np.random.randn(self.n_features, n_samples)
        logodds = self.x_to_logodds(X)
        probs = self.logodds_to_probs(logodds)
        y = np.zeros(n_samples)
        for ind_samples in range(n_samples):
            y[ind_samples] = np.random.choice(self.labels, p=probs[:, ind_samples])
        return SimData("OrdinalLR", X.T, y.astype(int), probs=probs).shuffle()

# BVv[Huul}#X9Yl5YH@(2Tj1xg

class SimLDA(object):
    def __init__(self, centroids, covariance):
        """
        Initialization.

        Parameters
        ----------
        centroids: centriods of classes, shape = [n_features, n_classes]

        covariance: shared covariance of multivariate Gaussian.

        sq_inv: sqrt(inv(covariance))
        """
        self.centroids = centroids
        self.covraiance = covariance
        self.n_features = centroids.shape[0]
        self.n_classes = centroids.shape[1]
        u, s, vh = linalg.svd(covariance)
        self.sq_inv = np.dot(u * np.sqrt(1 / s), vh)
    
    def gen(self, n_samples, seed=None):
        """
        Sampling.

        Parameters
        ----------
        n_samples: `int`, total number of samples; `array-like`, shape = [n_classes, ]

        Returns
        -------
        SimData(None, y, X)
            probs: None
            y: class labels, integer array
            X: covairates, shape = [n_features, n_samples]
        """
        if seed is not None: np.random.seed(seed)
        if isinstance(n_samples, int):
            n_samples_arr = np.zeros(self.n_classes, dtype=int) + int(n_samples / self.n_classes)
        else:
            n_samples_arr = n_samples.astype(int)
            assert(n_samples_arr.size == self.n_classes)
        X_list = []
        y_list = []
        for ind_class in range(self.n_classes):
            X = np.random.randn(self.n_features, n_samples_arr[ind_class])
            X = np.dot(self.sq_inv, X) + self.centroids[:, ind_class, np.newaxis]
            y = np.zeros(n_samples_arr[ind_class], dtype=int) + ind_class
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list, axis=1)
        y = np.concatenate(y_list)
        return SimData("LDA", X.T, y, self.centroids).shuffle()




if __name__ == "__main__":
    y_ori = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    transform = bidict({0:0, 1:0, 2:1})
    y_ext = extend_classes(y_ori, transform)
    coeff = np.array([0.5, -0.5, 1, -1])
    intercepts = np.array([5, 4.2, 3, -2])
    sim_ord = SimOrdLogit(5, coeff, intercepts)
    prob = np.array([.05, .05, .8, .05, .05])
    prob = prob[:, np.newaxis]
    x = sim_ord.probs_to_x(prob)
    logodds = sim_ord.probs_to_logodds(prob)
    prob_ = sim_ord.logodds_to_probs(logodds)
    print(sim_ord.logodds_to_probs(sim_ord.x_to_logodds(x)))

    centriods = np.random.randn(10, 5)
    covariance = np.eye(10)
    sim_lda = SimLDA(centriods, covariance)
    sim1 = sim_lda.gen(50)