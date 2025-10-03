import types
import pytest
from streamline.p3_feature_learning.utils.fl_loader import list_learners, load_learner

# pytest.skip("Tested Already", allow_module_level=True)

def test_list_learner_has_core_ids():
    imps = list_learners()
    # Expect at least the built-ins (adjust if your repo changes)
    for expected in ("pca",):
        assert expected in imps, f"Missing imputer id: {expected}"
        cls = imps[expected]
        assert isinstance(cls, type)
        assert hasattr(cls, "fit") and hasattr(cls, "transform") and hasattr(cls, "get_params") and hasattr(cls, "get_feature_names")

def test_load_learner_instantiates():
    lr = load_learner(learner_id="pca")
    print(lr)
    assert hasattr(lr, "fit")
    assert hasattr(lr, "transform")
    assert hasattr(lr, "get_feature_names")
    assert hasattr(lr, "get_params")
    
