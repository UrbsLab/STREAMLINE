# Phase 2: Imputation & Scaling — base interfaces (no typing, no shared base)

class Imputer:
    """
    Base interface for imputers.
    Contract:
      - fit(X, y, feature_meta) learns imputation statistics.
      - transform(X) applies them without reordering columns.
    """

    def __init__(self, component_id="imputer", random_state=None, **kwargs):
        self.id = component_id
        self.random_state = random_state
        self.params = dict(kwargs)

        # capability flags (override in subclasses or set via set_params)
        self.supports_nan_in_fit = True     # can learn with NaNs present
        self.preserves_dtype = False        # try to keep dtype if possible

    def get_params(self):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    # --- to implement ---
    def fit(self, X, y=None, feature_meta=None):
        raise NotImplementedError("Imputer.fit must be implemented")

    def transform(self, X):
        raise NotImplementedError("Imputer.transform must be implemented")


class Scaler:
    """
    Base interface for scalers.
    Contract:
      - fit(X, y, feature_meta) learns scaling params.
      - transform(X) applies them without reordering columns.
    """

    def __init__(self, component_id="scaler", random_state=None, **kwargs):
        self.id = component_id
        self.random_state = random_state
        self.params = dict(kwargs)

        # capability flags
        self.requires_dense = True               # most scalers need dense inputs
        self.scale_only_quantitative = True      # ignore categoricals by default

    def get_params(self):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    # --- to implement ---
    def fit(self, X, y=None, feature_meta=None):
        raise NotImplementedError("Scaler.fit must be implemented")

    def transform(self, X):
        raise NotImplementedError("Scaler.transform must be implemented")
