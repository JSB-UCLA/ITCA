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

   from itca import itca, compute_y_dist, bidict, GreedySearch
   # ===================  Inputs ========================
   # X, y_obs = dataset.features, dataset.ambigous_labels
   # =================== Evaluate s-ITCA ================
   combination = bidict({0:0, 1:0, 2:1})#combine class 0 and 1 into one
   itca(y_obs, y_pred, combination, compute_y_dist(y_obs))
   # ============= Search class combination =============
   gs = GreedySearch(class_type='ordinal')
   gs.search(X, y_obs, clf, verbose=False, early_stop=True)
   gs.selected # show the selected class combination

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

