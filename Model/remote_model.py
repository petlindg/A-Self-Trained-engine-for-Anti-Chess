from Model.model import Model


class RemoteModel(Model):
    def __init__(self, connection):
        self.connection = connection

    def eval(self, data):
        self.connection.send(('eval', data))
        return self.connection.recv()
