import threading
from collections import deque
import random
import numpy as np
from collections import deque
import tensorflow as tf
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import pickle


class BS:

    def __init__(self, users, rbs, id, env, state_size, action_size):
        self.loss = []
        self.state = []
        self.done = []
        self.id = id
        self.blocsNumber = len(rbs)
        self.users = users
        self.rbs = rbs
        self.connectedStations = []
        self.t = threading.Thread(name="start", target=self.start)
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=6000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.005
        self.model = self._build_model()
        self.fileName = 'res' + str(self.id) + '.pkl'
        self.fileName1 = 'eps' + str(self.id) + '.pkl'

    def connectToStation(self, bs):
        self.connectedStations.append(bs)

    def start(self):
        EPISODES = 1000
        episodes_time = []
        episodes_eps = []
        for _ in range(EPISODES):
            self.done.append(False)
        batch_size = 32
        for self.e in range(EPISODES):
            # print "BS ", self.id, ", episode : ", self.e
            self.state = []
            self.env.reset()
            for i in range(self.blocsNumber * self.env.stationsNumber):
                if self.blocsNumber * self.id <= i < self.blocsNumber * (self.id + 1):
                    self.state.append(0)
                else:
                    self.state.append(1)
            self.state = np.reshape(self.state, [1, len(self.state)])
            time = 0
            episode_reward = 0
            while not self.env.done(self.e) or time < 2000:
                if time >= 2000:
                    continue

                if time % 10 == 0:
                    for user in self.users:
                        user.neededAr += user.packetSize * user.pArravingRate

                action = self.act(self.state)
                next_state, reward, done = self.step(action, self.state)
                episode_reward += reward
                self.memorize(self.state, action, reward, next_state, done)
                self.state = next_state
                time += 1
                sleep(0.0001)
                # print("BS ", self.id, " score: ", self.time)

                if done:
                    # print("BS ", self.id, ", episode: ", self.e, " episode: ", episode_reward, " e: ", self.epsilon)
                    break

                if len(self.memory) > batch_size:
                    loss = self.replay(batch_size)
                    # Logging training loss every 10 timesteps
                    if time % 10 == 0:
                        print(self.e, time, loss)

            # print "BS ", self.id, ", episode : ", self.e, " , reward ", episode_reward
            episodes_time.append(time)
            episodes_eps.append(episode_reward)

        plt.figure(figsize=(9, 9))
        names = range(len(self.loss))
        values = self.loss
        plt.plot(names, values)
        plt.show()
        self.model.save("model.h5")
        self.file.close()

    def run(self):
        self.t.start()

    def join(self):
        self.t.join()

    def countAchievableRate(self, rb, usrId):
        gain = self.env.gain[rb.id][usrId][self.id]
        return rb.bandwidth * np.log2(1 + (rb.transmissionPower * gain) / rb.noisePower)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(24, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.action_size, activation='tanh'))
        self.model.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return self.model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])

        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        self.loss.append(loss)

        with open(self.fileName, 'ab') as handle:
            pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.fileName1, 'ab') as handle:
            pickle.dump((loss, self.epsilon), handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.id == 0 and self.e % 20 == 0:
            print('BS : ', self.id, ', episode : ', self.e, ' loss : ', loss, ' , epsilon : ', self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def step(self, action, state):
        reward = 0
        row = len(self.users)
        col = self.blocsNumber * len(self.env.stationsList)
        for i in range(row):
            for ii in range(col):
                if state[0][ii] == 0.5 or not action & (1 << (i * col + ii)):
                    continue
                rbId = ii

                if state[0][rbId] == 0:
                    rb = self.rbs[rbId % self.blocsNumber]
                    if rb.value == 0:
                        ar = self.countAchievableRate(rb, self.users[i].id)
                        reward += ar
                        self.users[i].neededAr -= ar
                        state[0][rbId] = 0.5
                        rb.value = 1
                        self.env.updateState()
                else:
                    rb = self.env.stationsList[int(rbId / self.blocsNumber)].rbs[rbId % self.blocsNumber]
                    if rb.value == 0:
                        borrowed = True
                        for rrb in self.rbs:
                            if rrb.value == 0:
                                borrowed = False
                        if borrowed:
                            ar = self.countAchievableRate(rb, self.users[i].id)
                            reward += ar
                            self.users[i].neededAr -= ar
                            state[0][rbId] = 0.5
                            rb.value = 1
                            self.env.updateState()

        ddone = True
        for rb in self.rbs:
            if rb.value == 0:
                ddone = False
        self.done[self.e] = ddone

        return state, reward, self.env.done(self.e)
