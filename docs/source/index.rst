.. itca documentation master file, created by
   sphinx-quickstart on Sun Aug 15 20:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ITCA's documentation!
================================

**ITCA**  (Information-theoretic classification accuracy) is a criterion that guides data-driven combination of ambiguous outcome labels in multi-class classification.


Installation
------------

Requirements:

- python >= 3.6
- numpy: https://pypi.org/project/numpy/
- scikit-learn: https://pypi.org/project/scikit-learn/

Install from source code:

.. code:: shell

   git clone https://github.com/JSB-UCLA/ITCA.git
   cd ITCA
   python setup.py install

ITCA is easy to use.

.. code:: python

   import numpy as np
   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
   from itca import itca, compute_y_dist, bidict, GreedySearch
   # ===================  Data ========================
   # `X` is the feature matrix, a 2D numpy array of size (n_smaples, n_features).
   # `y_obs` is the observed labels, a 1D numpy array of size (n_samples, ) that takes values 
   # in [0, 1, 2] (the observed classes number K0=3).
   true_combination = bidict({0:0, 1:0, 2:1, 3:2})
   X1 = np.array([[0., 0.]]) + np.random.randn(200,  2)
   X2 = np.array([[1.5, 1.5]]) + np.random.randn(200, 2)
   X3 = np.array([[-1.5, 1.5]]) + np.random.randn(200, 2)
   X = np.concatenate([X1, X2, X3])  # data matrix
   y_true = np.concatenate([np.ones(200) * i for i in 
   range(3)]).astype(int)            # true lables K^*=3
   y_obs = true_combination.reverse_map(y_true) # observed labels K_0=4
   #===========  Classsification algorithm =============
   # `clf` can be any sklearn classifcation algorithm or any classifcation algorithm that implements 
   # `clf.fit(X, y)` for fitting and `clf.predict(X)` for prediction.  
   clf = LinearDiscriminantAnalysis()
   # =================== Evaluate s-ITCA ================
   clf.fit(X, true_combination.map(y_obs))
   y_pred = clf.predict(X)
   itca(y_obs, y_pred, true_combination, compute_y_dist(y_obs))
   # ============= Search class combination =============
   gs = GreedySearch(class_type='ordinal')
   gs.search(X, y_obs, clf, verbose=False, early_stop=True)
   gs.selected # show the selected class combination
   #>>>{0: 0, 1: 0, 2: 1, 3: 2}|ITCA=0.8807|

Please see the [tutorial](https://messcode.github.io/ITCA/tutorials.htmll)  for more details.


Contribute
----------
- Issue tracker:  https://github.com/messcode/ITCA/issues
- Source code:
	- https://github.com/JSB-UCLA/ITCA
	- https://github.com/messcode/ITCA (the devlopmental version)

Contact
-------
If you are having any issues, comments regarding this project, please feel free to contact zhang.dabiao11@gmail.com


.. toctree::
   :caption: Contents
   :maxdepth: 1

   installation
   tutorials
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

