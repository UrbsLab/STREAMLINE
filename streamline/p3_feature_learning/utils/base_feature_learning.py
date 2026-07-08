# Phase 3: Feature Learning — base interface (no typing, flags on self)

class FeatureLearner(object):
    """
    Base interface for optional feature learning / transformation steps.
    Examples: PCA/ICA/NMF, polynomial features, random features, FIBERS, etc.

    Contract:
      - fit(X, y, feature_meta) learns transform parameters.
      - transform(X) applies them; must preserve row order.
      - get_feature_names_out(input_features) returns names for output columns.
      - get_parent_map(output_features) maps produced features -> source columns.
    """

    def __init__(self, component_id="feature_learner", random_state=None, **kwargs):
        # identifiers & params
        self.id = component_id
        self.random_state = random_state
        self.params = dict(kwargs)

        # capability flags (override in subclasses or via set_params)
        self.needs_quantitative = False    # True if input must be numeric-only
        self.is_supervised = False         # True if y is required during fit
        self.produces_sparse = False       # True if transform returns sparse

    # ---------- lifecycle ----------
    def get_params(self):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self
    
    # ---------- fit/transform ----------
    def fit(self, X, y=None, feature_meta=None):
        raise NotImplementedError("FeatureLearner.fit must be implemented")

    def transform(self, X):
        raise NotImplementedError("FeatureLearner.transform must be implemented")

    def fit_transform(self, X, y=None, feature_meta=None):
        self.fit(X, y, feature_meta)
        return self.transform(X)

    # ---------- names & lineage ----------
    def get_feature_names_out(self, input_features):
        """
        Return names for columns produced by transform().
        Default: identity (no change).
        """
        return list(input_features)

    def get_parent_map(self, output_features):
        """
        Map each produced feature -> list of parent input feature names.
        Default: identity mapping.
        """
        return dict((name, [name]) for name in output_features)
