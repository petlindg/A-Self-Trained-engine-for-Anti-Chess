class Connection:
    def __init__(self, connection_id, outgoing_queue, incoming_queue):
        self.connection_id = connection_id
        self.outgoing_queue = outgoing_queue
        self.incoming_queue = incoming_queue

    def send(self, data):
        self.outgoing_queue.put((self.connection_id, data))

    def receive(self):
        return self.incoming_queue.get()
