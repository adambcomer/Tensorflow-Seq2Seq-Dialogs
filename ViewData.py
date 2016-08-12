import matplotlib.pyplot as pyplot
import numpy as np

data = np.loadtxt(open("/Users/adamcomer/PycharmProjects/TensorflowSeq/LogLoss.csv", "rb"), delimiter=",", skiprows=0)

print(data)

pyplot.plot([x[0] for x in data], [x[1] for x in data])

pyplot.show()