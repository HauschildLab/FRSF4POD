from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
import numpy as np
from typing import Literal, Optional, Self
from numpy.typing import ArrayLike
import pandas as pd
from .persistence import SaveLoadMixin
from sksurv.functions import StepFunction


class LocalRandomSurvivalForest(RandomSurvivalForest, SaveLoadMixin):
    """A local random survival forest.

    A local random survival forest is a meta estimator that fits a number of
    survival trees on various sub-samples of the dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample
    size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    It is an extension of the random survival forest described in [1]_ and [2]_,
    which is a generalization of the random forest method to
    analyze right-censored survival data.

    This extension provides the ability to update the local model
    with trees trained on other sites,
    which can be used in a federated learning setting.
    The local model can switch between using its own trees and the federated trees
    based on the specified update method and weighting scheme.

    In each survival tree, the quality of a split is measured by the
    log-rank splitting rule.

    See the :ref:`User Guide </user_guide/random-survival-forest.ipynb>`,
    [1]_ and [2]_ for further description.

    Parameters
    ----------
    update_method: str Literal "constant" or "all", default: "all"
        The method to use when updating the local model with federated trees.
            - "constant": maintain the same number of trees as specified
            by `n_estimators`by randomly sampling from the federated trees
            according to the specified weighting scheme.
            - "all": use all federated trees,
            which may result in a different number of trees
            than specified by `n_estimators`.

    update_weighting: str Literal "equal" or "site_size", default: "equal",
        The weighting scheme to use when sampling federated trees for the "constant" update method.
            - "equal": sample federated trees with equal probability.
            - "site_size": sample federated trees with probability
            proportional to the size of the site that trained the tree.

    n_estimators : int, optional, default: 100
        The number of trees in the forest.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {'sqrt', 'log2'} or None, optional, default: 'sqrt'
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : bool, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, optional, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional, default: None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, optional, default: None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    low_memory : bool, optional, default: False
        If set, :meth:`predict` computations use reduced memory but :meth:`predict_cumulative_hazard_function`
        and :meth:`predict_survival_function` are not implemented.

    Attributes
    ----------

    site_size : int
        The number of samples in the local dataset.
        Available after fitting the model.

    local_features : set
        The set of features that are available in the local dataset.
        Available after fitting the model.

    all_features : list
        The list of all features in the order they are used in the model.
        Available after fitting the model.

    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.

    unique_times_ : ndarray, shape = (n_unique_times,)
        Unique time points.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.

    Methods
    -------
    set_federated_estimators(estimators)
        Sets the federated estimators to be used for updating the local model.

    use_federal_estimators()
        Switches the local model to use trees trained on federated sites.

    use_local_estimators()
        Switches the local model to use only the trees trained locally.

    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    Compared to scikit-learn's random forest models, :class:`RandomSurvivalForest`
    currently does not support controlling the depth of a tree based on the log-rank
    test statistics or it's associated p-value, i.e., the parameters
    `min_impurity_decrease` or `min_impurity_split` are absent.
    In addition, the `feature_importances_` attribute is not available.
    It is recommended to estimate feature importances via
    :func:`sklearn.inspection.permutation_importance`.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behavior during
    fitting, ``random_state`` has to be fixed.

    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841-860.

    .. [2] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25-31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    def __init__(
        self,
        update_method: Literal["constant", "all"] = "all",
        update_weighting: Literal["equal", "site_size"] = "equal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_features: set = set()
        self._federated_estimators: list[SurvivalTree] = []
        self.site_size: int = 0
        self.update_method = update_method
        self.update_weighting = update_weighting
        self.tree_origin: Literal["local", "federated"] = "local"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Self:
        """
        Build a forest of survival trees from the training set (X, y).
        Stores metadata about the local dataset
        such as the number of samples and the features available,
        which can be used for updating the model with federated trees.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.
        """
        self.site_size: int = len(X)
        self.local_features = set(X.columns[~X.isna().all()].tolist())
        self.all_features = X.columns.tolist()
        super().fit(X, y, sample_weight=sample_weight)
        for estimator in self.estimators_:
            estimator.sample_weight = self.site_size / self.n_estimators

        return self

    def predict_survival_function_custom(self, X, return_array=False):
        if self.tree_origin == "local":
            unique_times = self.unique_times_
        else:
            unique_times = self.federated_unique_times

        preds = []

        for estimator in self.estimators_:
            pred = estimator.predict_survival_function(X)
            preds.append(pred)

        stepfunctions = []
        for i in range(len(X)):
            Ys = []
            for pred in preds:
                X = pred[i].x
                Y = pred[i].y
                indexes = np.searchsorted(X, unique_times, side="right")
                Y_extended = np.array([1] + list(Y) + [0])

                Y_fed = Y_extended[indexes]
                Ys.append(Y_fed)

            y_hat = sum(Ys) / len(self.estimators_)

            sf = StepFunction(x=unique_times, y=y_hat, a=preds[0][i].a, b=preds[0][i].b)

            stepfunctions.append(sf)

        return np.array(stepfunctions)

    def set_federated_estimators(self, estimators: list[SurvivalTree]) -> Self:
        """
        Sets the federated estimators to be used for updating the local model.
        This method is intended to be called by the federated model when distributing the federated trees to the local models.

        Parameters
        ----------
        estimators : list of SurvivalTree instances
            The federated estimators to be used for updating the local model.
        """

        self._federated_estimators = estimators
        return self

    def use_local_estimators(self) -> Self:
        """
        Switches the local model to use only the trees trained locally.
        This method can be used to switch back to the local trees after
        using the federated trees, which can be useful for evaluating
        the performance of the local model
        before and after updating with federated trees.
        """
        if self.tree_origin == "local":
            return self

        self.tree_origin = "local"

        self.estimators_ = self._local_estimators
        self.n_estimators = len(self.estimators_)
        return self

    def use_federated_estimators(self, random_state: Optional[int] = None) -> Self:
        """
        Switches the local model to use trees trained on federated sites.
        This method can be used to update the local model with federated trees,

        Parameters
        ----------
        random_state : int, optional, default: None
            The random state to use for sampling federated trees when the "constant" update method is used.
            This parameter is only relevant when `update_method` is set to "constant".
            If None, the random state will be determined by the `random_state` attribute of the model.
        """
        if self.tree_origin == "federated":
            return self

        if random_state is None:
            random_state = self.random_state

        self._local_estimators = self.estimators_
        self.tree_origin = "federated"

        if self.update_method == "all":
            self.estimators_ = self._federated_estimators

        elif self.update_method == "constant":
            if self.update_weighting == "equal":
                weights: list[float] = [1.0] * len(self._federated_estimators)
            elif self.update_weighting == "site_size":
                weights: list[float] = [
                    estimator.sample_weight for estimator in self._federated_estimators
                ]
            self.estimators_ = np.random.default_rng(random_state).choice(
                self._federated_estimators,
                size=self.n_estimators,
                p=np.array(weights) / sum(weights),
            )

        federated_unique_times = set()
        for estimator in self.estimators_:
            federated_unique_times = federated_unique_times.union(
                set(estimator.unique_times_)
            )

        self.federated_unique_times = np.sort(np.array(list(federated_unique_times)))
        self.n_estimators = len(self.estimators_)
        return self


class FederatedRandomSurvivalForest(RandomSurvivalForest, SaveLoadMixin):
    """A federatedrandom survival forest.

    A federated random survival forest is a helper class
    that aggregates and distributes the estimators of
    multiple local random survival forests.

    This class is not intended to be fit or used for prediction directly.
    Instead, it serves as a container for the local models and their trees.
    It provides methods to distribute the federated trees to the local models
    based on the features used by each tree and the features available locally.

    See the :ref:`User Guide </user_guide/random-survival-forest.ipynb>`,
    [1]_ and [2]_ for further description.

    Parameters
    ----------
    local_models : list of LocalRandomSurvivalForest instances
        The local random survival forests to aggregate.

    Attributes
    ----------
    local_models : list of LocalRandomSurvivalForest instances
        The local random survival forests that are aggregated.

    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators from all local models.

    n_estimators : int
        The total number of trees in the federated model, which is
        the sum of the number of trees in all local models.

    tree_features: list of sets
        The features used by each tree in estimators_.

    Methods
    -------
    add_local_model(local_model)
        Adds a local random survival forest to the federated model and
        updates the collection of estimators and their features.

    distribute_trees()
        Distributes the federated trees to the local models based on
        the features used by each tree and the features available locally.

    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.

    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841-860.

    .. [2] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25-31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    def __init__(
        self,
        local_models: list[LocalRandomSurvivalForest],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.all_features: np.ndarray = local_models[0].all_features

        self.local_models: list[LocalRandomSurvivalForest] = []
        self.estimators_: list[SurvivalTree] = []
        self.tree_features: list[set] = []
        for model in local_models:
            self.add_local_model(model)

    def add_local_model(self, local_model: LocalRandomSurvivalForest) -> Self:
        """
        Adds a local random survival forest to the federated model and
        updates the collection of estimators and their features.

        This method can be used to incrementally add local models to the federated model,
        which can be useful in a dynamic federated learning setting where new clients may join over time.

        Parameters
        ----------
        local_model : LocalRandomSurvivalForest
            The local random survival forest to add to the federated model.
        """
        local_tree_features = []
        for estimator in local_model.estimators_:
            tree_features = estimator.tree_.feature
            tree_features_names = set(
                [self.all_features[i] for i in tree_features if i >= 0]
            )
            local_tree_features.append(tree_features_names)

        self.local_models.append(local_model)
        self.estimators_.extend(local_model.estimators_)
        self.tree_features.extend(local_tree_features)
        self.n_estimators = len(self.estimators_)
        return self

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Federated model cannot be fit directly.")

    def predict(self, X):
        raise NotImplementedError("Federated model cannot predict directly.")

    def distribute_trees(self) -> list[LocalRandomSurvivalForest]:
        """
        Distributes the federated trees to the local models based on
        the features used by each tree and the features available locally.
        Each local model will receive the trees that only use features available locally.
        This method should be called after all local models have been added to the federated model.
        """
        for model in self.local_models:
            valid_estimators = []
            for estimator, feat_set in zip(self.estimators_, self.tree_features):
                # find all estimators that only use features available locally
                if feat_set.issubset(model.local_features):
                    valid_estimators.append(estimator)

            model.set_federated_estimators(valid_estimators)

        return self.local_models
