from User import User


class EmbbUser(User):
    def __init__(self, id):
        User.__init__(self, id)
        self.rMin = 100
        self.dMax = 100
        self.packetSize = 400
