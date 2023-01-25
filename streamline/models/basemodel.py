from sklearn import clone
from streamline.modeling.parameters import get_parameters


class MLModel:
    def __int__(self):
        self.model = None
        self.model_name = None
        self.param_grid = get_parameters(self.model_name)
        self.model = clone(self.model).set_params(**self.param_grid)

    def objective(self):
        pass

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x_in):
        self.model.predict(x_in)
