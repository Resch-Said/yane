from copy import deepcopy


class Connection:
    ID = 0

    def __init__(self):
        self.weight = 1.0
        self.in_neuron = None
        self.out_neuron = None
        self.enabled = True
        self.id = Connection.ID
        Connection.ID += 1

    def set_weight(self, weight):
        self.weight = weight

    def set_in_neuron(self, neuron):
        self.in_neuron = neuron

    def set_out_neuron(self, neuron):
        self.out_neuron = neuron

    def set_enabled(self, enabled):
        self.enabled = enabled

    def get_weight(self):
        return self.weight

    def get_in_neuron(self):
        return self.in_neuron

    def get_out_neuron(self):
        return self.out_neuron

    def is_enabled(self):
        return self.enabled

    def __str__(self):
        return "Connection: " + str(self.id) + " from " + str(self.in_neuron.id) + " to " + str(
            self.out_neuron.id) + " with weight " + str(self.weight) + " and enabled: " + str(self.enabled)

    def get_id(self):
        return self.id

    def copy(self):
        return deepcopy(self)
