import numpy as np

from BS import BS
from random import randrange
from math import pow
from EmbbUser import EmbbUser
from RB import RB
from UrllcUser import UrllcUser


class Env:
    def __init__(self):
        self.stationsNumber = 2
        self.usersNumber = 6
        self.urllcUsersNumber = 4
        self.embbUsersNumber = 2
        self.rbNumber = 5
        self.stationsList = []

        currentUserIndice = 0
        for i in range(0, self.stationsNumber):
            currentRbId = 0
            usersList = []
            if i == self.stationsNumber - 1:
                uUserNumber = self.urllcUsersNumber
                eUserNumber = self.embbUsersNumber
            else:
                uUserNumber = self.urllcUsersNumber / (self.stationsNumber - i)
                self.urllcUsersNumber = self.urllcUsersNumber - uUserNumber
                eUserNumber = self.embbUsersNumber / (self.stationsNumber - i)
                self.embbUsersNumber = self.embbUsersNumber - eUserNumber

            for _ in range(0, uUserNumber):
                usersList.append(UrllcUser(currentUserIndice))
                currentUserIndice += 1

            for _ in range(0, eUserNumber):
                usersList.append(EmbbUser(currentUserIndice))
                currentUserIndice += 1

            rbList = []
            for _ in range(0, self.rbNumber):
                rbList.append(RB(currentRbId))
                currentRbId += 1
            totalrbNumber = self.stationsNumber * self.rbNumber
            print int(pow(2, len(usersList) * totalrbNumber))
            self.stationsList.append(
                BS(usersList, rbList, i, self, totalrbNumber, long(pow(2, len(usersList) * totalrbNumber))))

        xyUEs = np.vstack((np.random.uniform(low=0.0, high=500, size=self.usersNumber),
                           np.random.uniform(low=0.0, high=500, size=self.usersNumber)))
        xyBSs = np.vstack((np.random.uniform(low=0.0, high=500, size=self.stationsNumber),
                           np.random.uniform(low=0.0, high=500, size=self.stationsNumber)))
        self.gain = channel_gain(self.usersNumber, self.stationsNumber, self.rbNumber, xyUEs, xyBSs, 0)
        for i in range(0, self.stationsNumber):
            for j in range(0, self.stationsNumber):
                if i != j:
                    self.stationsList[i].connectToStation(self.stationsList[j])
            self.stationsList[i].run()
        for i in range(self.stationsNumber):
            self.stationsList[i].join()

    def reset(self):
        for bs in self.stationsList:
            for rb in bs.rbs:
                rb.value = 0
            for user in bs.users:
                user.neededAr = 0

    def done(self, e):
        for bs in self.stationsList:
            if len(bs.done) <= e or not bs.done[e]:
                return False
        return True

    def updateState(self):
        id = 0
        for bs in self.stationsList:
            for rb in bs.rbs:
                if rb.value == 1 and len(bs.state) > id:
                    bs.state[0][id] = 0.5

                id += 1



def channel_gain(K, N, C, xyUEs, xySBSs, xi):
    plSBSs = np.zeros((K, N))  # Path-loss between users and the SBSs
    # print(plSBSs)
    for k in range(K):
        for n in range(N):
            # Shadowing
            # Xi = np.random.normal(0.0, 8.0)
            plSBSs[k, n] = 10 ** (
                    -3.06 - 3.67 * np.log10(np.linalg.norm(xyUEs[:, k] - xySBSs[:, n])) + xi / 10.0)
    h = np.random.exponential(scale=1.0, size=(C, K, N))
    g = np.multiply(h, plSBSs)
    return g


env = Env()
