import pytest
from streamline.modeling.load_models import load_class_from_folder

# pytest.skip("Tested Already", allow_module_level=True)

actual = ['Elastic Net',
          'ExSTraCS',
          'XCS',
          'eLCS',
          'Support Vector Machine',
          'Category Gradient Boosting',
          'Gradient Boosting',
          'Light Gradient Boosting',
          'Extreme Gradient Boosting',
          'Logistic Regression',
          'Naive Bayes',
          'Decision Tree',
          'K-Nearest Neighbors',
          'Artificial Neural Network',
          'Random Forest',
          'Genetic Programming']


def test_load_class_from_folder():
    loaded = load_class_from_folder()
    loaded = [m.model_name for m in loaded]
    assert (loaded == actual)
    loaded = load_class_from_folder("Classification")
    loaded = [m.model_name for m in loaded]
    assert (loaded == actual)
    loaded = load_class_from_folder("Regression")
    loaded = [m.model_name for m in loaded]
    assert (loaded == ["Support Vector Regression"])
