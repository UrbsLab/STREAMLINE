class BaseMLModel:
    def __int__(self):
        self.model = None
        self.model_name = None
        self.param_grid = None

        pass

    def fit(self, x, y):
        pass

    def predict(self, x_in):
        pass
