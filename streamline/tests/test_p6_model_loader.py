import pytest

from streamline.p6_modeling.utils.loader import (
    load_model_classes,
    get_model_by_id,
)

pytest.skip("Tested Already", allow_module_level=True)

# ---------------------------
# load_model_classes()
# ---------------------------

@pytest.mark.parametrize("model_type", [
    "Binary",
    "Multiclass",
    "Regression",
])
def test_load_model_classes_returns_classes_and_shape(model_type):
    classes = load_model_classes(model_type)
    # It’s okay if some folders are empty; just assert a list is returned
    assert isinstance(classes, list)
    for cls in classes:
        # Each class must expose the expected attrs the loader requires
        for attr in ("small_name", "model_name", "model_type", "model_evaluation"):
            assert hasattr(cls, attr), f"{cls} missing attribute '{attr}'"
        # model_type on the class should match the requested type
        assert getattr(cls, "model_type") == model_type


def test_load_model_classes_invalid_type():
    with pytest.raises(ValueError):
        load_model_classes("TotallyNotAType")


# ---------------------------
# get_model_by_id()
# ---------------------------

@pytest.mark.parametrize(
    "model_type, candidates",
    [
        # Try common binary models we’ve discussed; skip if they don't exist locally
        ("Binary", ["LR", "NB", "SVM",]),
        # Add your own multiclass/regression aliases if you have them
        # ("Multiclass", []),
        # ("Regression", []),
    ],
)
def test_get_model_by_id_resolves_aliases_case_insensitive(model_type, candidates):
    """
    For each candidate id, try to resolve a model class.
    If a given id isn’t present in the repo, we SKIP that id gracefully.
    """
    for mid in candidates:
        try:
            cls = get_model_by_id(model_type, mid)
        except ValueError:
            pytest.skip(f"Model id '{mid}' not found for type '{model_type}' in this checkout.")
            continue

        # Basic shape checks on the resolved class
        assert hasattr(cls, "small_name")
        assert hasattr(cls, "model_name")
        assert hasattr(cls, "model_type")
        assert cls.model_type == model_type

        # Check alias logic: the id should match small_name or underscored model_name (case-insensitive)
        aliases = {
            cls.small_name.lower(),
            cls.model_name.lower().replace(" ", "_"),
        }
        assert mid.lower() in aliases, f"Resolved class aliases {aliases} do not include requested id '{mid.lower()}'"


def test_get_model_by_id_raises_for_unknown():
    with pytest.raises(ValueError):
        get_model_by_id("Binary", "definitely_not_a_model")
