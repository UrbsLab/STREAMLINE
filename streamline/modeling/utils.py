from streamline.models.artificial_neural_network import MLPClassifier
from streamline.models.decision_tree import DecisionTreeClassifier
from streamline.models.genetic_programming import GPClassifier
from streamline.models.gradient_boosting import GBClassifier
from streamline.models.gradient_boosting import XGBClassifier
from streamline.models.gradient_boosting import CGBClassifier
from streamline.models.gradient_boosting import LGBClassifier
from streamline.models.learning_based import eLCSClassifier
from streamline.models.learning_based import XCSClassifier
from streamline.models.learning_based import ExSTraCSClassifier
from streamline.models.linear_model import LogisticRegression
from streamline.models.naive_bayes import NaiveBayesClassifier
from streamline.models.neighbouring import KNNClassifier
from streamline.models.random_forest import RandomForestClassifier
from streamline.models.support_vector_machine import SupportVectorClassifier

SUPPORTED_MODELS = [
    'Naive Bayes',
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'Extreme Gradient Boosting',
    'Light Gradient Boosting',
    'Category Gradient Boosting',
    'Support Vector Machine',
    'Artificial Neural Network',
    'K-Nearest Neighbors',
    'Genetic Programming',
    'eLCS',
    'XCS',
    'ExSTraCS'
]

SUPPORTED_MODELS_SMALL = ['NB', 'LR', 'DT', 'RF', 'GB', 'XGB', 'LGB', 'CGB', 'SVM', 'ANN', 'KNN',
                          'GP', 'eLCS', 'XCS', 'ExSTraCS']

SUPPORTED_MODELS_OBJ = [
    NaiveBayesClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBClassifier,
    XGBClassifier,
    LGBClassifier,
    CGBClassifier,
    SupportVectorClassifier,
    MLPClassifier,
    KNNClassifier,
    GPClassifier,
    eLCSClassifier,
    XCSClassifier,
    ExSTraCSClassifier,
]

MODEL_DICT = dict(zip(SUPPORTED_MODELS + SUPPORTED_MODELS_SMALL,
                      SUPPORTED_MODELS_OBJ + SUPPORTED_MODELS_OBJ))


def is_supported_model(string):
    return string in SUPPORTED_MODELS or string in SUPPORTED_MODELS_SMALL


def model_str_to_obj(string):
    assert is_supported_model(string)
    return MODEL_DICT[string]
