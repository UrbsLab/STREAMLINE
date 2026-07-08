from collections import defaultdict
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import _BaseComposition

class _BaseXComposition(_BaseComposition):
    """
    parameter handler for list of estimators
    """

    def _set_params(self, attr, named_attr, **params):
        # Ordered parameter replacement
        # 1. root parameter
        if attr in params:
            setattr(self, attr, params.pop(attr))

        # 2. single estimator replacement
        items = getattr(self, named_attr)
        names = []
        if items:
            names, estimators = zip(*items)
            estimators = list(estimators)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                # replace single estimator and re-build the
                # root estimators list
                for i, est_name in enumerate(names):
                    if est_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del estimators[i]
                        else:
                            estimators[i] = new_val
                        break
                # replace the root estimators
                setattr(self, attr, estimators)

        # 3. estimator parameters and other initialisation arguments
        super(_BaseXComposition, self).set_params(**params)
        return self

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator. """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this method."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {"name": type(estimator).__name__})
    
def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for _, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))

