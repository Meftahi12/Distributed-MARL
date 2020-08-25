import pickle

import matplotlib.pyplot as plt
objects = []
with (open("res1.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break



plt.figure(figsize=(9,9))
names = range(len(objects))
values = objects
plt.plot(names, values)
plt.show()
