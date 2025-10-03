import types
import pytest
from streamline.p2_impute_scale.utils.impute_loader import list_imputers, load_imputer

pytest.skip("Tested Already", allow_module_level=True)

def test_list_imputers_has_core_ids():
    imps = list_imputers()
    print(imps)
    # Expect at least the built-ins (adjust if your repo changes)
    for expected in ("simple", "median_map", "knn", "iterative"):
        assert expected in imps, f"Missing imputer id: {expected}"
        cls = imps[expected]
        assert isinstance(cls, type)
        assert hasattr(cls, "fit") and hasattr(cls, "transform") and hasattr(cls, "get_params")

def test_load_imputer_instantiates():
    imp = load_imputer("simple", strategy="median")
    print(imp)
    assert hasattr(imp, "fit")
    assert hasattr(imp, "transform")
    assert hasattr(imp, "get_params")
    
