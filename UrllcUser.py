from User import User


class UrllcUser (User):
    def __init__(self, id):
        User.__init__(self, id)
        self.rMin = 10
        self.dMax = 10
        self.packetSize = 120