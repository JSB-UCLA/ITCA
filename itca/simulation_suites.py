from itca.metrics import itca, prediction_entropy
import numpy as np

from functools import reduce
from itca.simulator import SimLDA, SimOrdLogit, SimData

def lda_suite(n=1000, p=5, n_classes=5, length=2, seed=0, min_dist=.5):
    np.random.seed(seed)
    assert(min_dist < length * .8)
    def gen_centroids(p, n_classes):
        centroids = np.zeros([p, n_classes])
        count = 0
        for ind in range(1, n_classes):
            d = np.random.randn(p)
            while True:
                next_centriod = centroids[:, ind - 1] + length * d  / np.linalg.norm(d)
                centroids_dist = np.sqrt(np.sum((centroids[:, :ind] - next_centriod[:, None])**2, axis=0))
                if np.min(centroids_dist) > min_dist:
                    centroids[:, ind] = next_centriod
                    break
                count += 1
                if count > 100:
                    return None
        return centroids
    while True:
        centroids = gen_centroids(p, n_classes)
        if centroids is not None:
            break
        print("Regenerate data.")
    cov = np.eye(p)
    sim_lda = SimLDA(centroids, cov)
    sim = sim_lda.gen(n)
    return sim

def olr_suite(n=1000, p=5, n_classes=5, majority=.8, seed=0):
    np.random.seed(seed)
    sim_list = []
    f = lambda sim1, sim2: sim1.concat(sim2)
    coeff = np.array([.5, -.5, .5, 1, -1, ])
    intercepts = np.linspace(1, -1, n_classes - 1)
    sim_olr = SimOrdLogit(n_classes, coeff, intercepts)
    for ind in range(n_classes):
        prob = np.zeros(n_classes) + (1 - majority) / (n_classes - 1)
        prob[ind] = majority
        prob = prob[:, np.newaxis]
        xi = sim_olr.probs_to_x(prob)
        ni = int(n / n_classes)
        X = np.random.randn(p, ni) + xi
        simi = sim_olr.gen(ni, X=X)
        sim_list.append(simi)
    return reduce(f, sim_list)

def eval_model(sim, model, metric, prop=.1):
    test_size = int(prop * sim.size)
    sim_test = sim[:test_size]
    sim_train = sim[test_size:]
    model.fit(sim_train.X.T, sim_train.y)
    y_pred = model.predict(sim_test.X.T)
    return metric(sim_test.y, y_pred)

def eval_model_transform(sim, model, metric, transform, prop=.1, 
has_transform=False, y_dist=None, y_ori_pred=None):
    test_size = int(prop * sim.size)
    sim_test = sim[:test_size]
    sim_train = sim[test_size:]
    model.fit(sim_train.X.T, transform.map(sim_train.y))
    y_pred = model.predict(sim_test.X.T)
    if y_ori_pred is not None:
        return metric(sim_test.y, y_ori_pred, y_pred, transform)
    if y_dist is not None:
        return metric(sim_test.y, y_pred, y_dist, transform)
    if has_transform:
        return metric(sim_test.y, y_pred, transform)
    else:
        return metric(transform.map(sim_test.y), y_pred)

if __name__ == "__main__":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from comb.utils import bidict
    # initialize model
    clf = LinearDiscriminantAnalysis()
    # mlr = LogisticRegression(penalty="l2", solver="lbfgs", 
    #                        multi_class="multinomial", C=10.0)
    # # generate simulation data
    # ext_tf = bidict({0:0, 1:0, 2:1, 3:2, 4:3, 5:4})
    # sim1 = lda_suite()
    # sim2 = olr_suite()
    # sim1_ext = SimData(sim1.generator_name, sim1.X, ext_tf.reverse_map(sim1.y))
    # sim2_ext = SimData(sim2.generator_name, sim2.X, ext_tf.reverse_map(sim2.y))
    # n_classes_ = [5, 4, 3, 2]
    # i = 0
    # for tf in enumerate_transforms(6, 3):
    #     print(tf)
    #     i = i + 1
    # print(i)
    # eval_model_transform(sim1_ext, clf, accuracy_score, tf)
    from comb.search import GreedySearch, BFSearch
    gs1 = GreedySearch(class_type="ordinal", metric=itca)
    gs2 = GreedySearch(class_type="ordinal", metric=prediction_entropy)
    sim1 = lda_suite(n=1000, p=2)
    ext_tf = bidict({0:0, 1:0, 2:1, 3:2, 4:3, 5:4, 6:4})
    sim1_ext = SimData(sim1.generator_name, sim1.X, ext_tf.reverse_map(sim1.y))
    # gs1.search(sim1_ext.X.T, sim1_ext.y, clf, verbose=True)
    # gs2.search(sim1_ext.X.T, sim1_ext.y, clf, verbose=True)
    # def print_root(root):
    #     print(root)
    #     if root.children:
    #         for child in root.children:
    #             print_root(child)
    # print_root(gs1.path)
