import numpy as np
import random

csv = np.loadtxt("/home/adam/TensorflowRNN/rebuiltTrainingData.csv", delimiter=",", dtype="float", skiprows=0)

train = []
test = []

for row in csv:
    if(random.random() <= 0.2):
        test.append(row)
    else:
        train.append(row)

train = np.array(train)
test = np.array(test)

np.savetxt("/home/adam/TensorflowRNN/rebuiltTrainingDataTrain.csv", train, delimiter=",")
np.savetxt("/home/adam/TensorflowRNN/rebuiltTrainingDataTest.csv", test, delimiter=",")