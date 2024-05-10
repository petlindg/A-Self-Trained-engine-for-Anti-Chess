from Model.model import Model


class LocalModel(Model):
    def __init__(self, model):
        self.model = model

    def eval(self, data):
        return self.model.predict(data, verbose=None)
