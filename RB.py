class RB:
    def __init__(self, id):
        self.transmissionPower = 1
        self.bandwidth = 180
        self.noisePower = 4e-18
        self.value = 0
        self.id = id

    def __str__(self):
        return str(self.id)