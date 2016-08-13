import tensorflow as tf
import numpy as np
import time
import re
from itertools import izip
batch_size = 32
seq_size = 75
x = tf.placeholder(tf.int32, shape=[None, None])
y = tf.placeholder(tf.int32, shape=[None, None])
targets = tf.placeholder(tf.int32, shape=[None, None])
drop = tf.placeholder("float")

dictionary = np.loadtxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/dictionary.csv", delimiter=" ", dtype="string", skiprows=0)
dictsize = dictionary.shape[0]

count = []

for i in range(dictsize):
    count.append(i)

dictionary = dict(izip(list(dictionary), count))
rvsdictionary = dict(izip(dictionary.values(), dictionary.keys()))
print(dictionary)

filename_queue = tf.train.string_input_producer(["/Users/adamcomer/PycharmProjects/TensorflowSeq/reviewed_dialogs.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[""], [""]]
col1, col2 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.pack(col1)
labels = tf.pack(col2)

teminp = []
temoutput = []
temtarget = []

for o in range(seq_size):
    teminp.append(x[:, o])
    temoutput.append(y[:, o])
    temtarget.append(targets[:, o])

W1 = tf.Variable(tf.truncated_normal([batch_size, seq_size], stddev=0.1))
W1_0 = []
for j in range(seq_size):
    W1_0.append(W1[:, j])

cell1 = tf.nn.rnn_cell.GRUCell(64)
#droplayer = tf.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=drop, output_keep_prob=drop)
cell = tf.nn.rnn_cell.MultiRNNCell([cell1] * 2)

rnn, state = tf.nn.seq2seq.embedding_attention_seq2seq(teminp, temoutput, cell, dictsize, dictsize, 100, feed_previous=True)

logits = tf.nn.seq2seq.sequence_loss(rnn, temtarget, W1_0)

train = tf.train.AdagradOptimizer(0.1).minimize(logits)

saver = tf.train.Saver()

init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

total = 0
rows = 1000
trainiterations = int(rows / batch_size)

saver.restore(sess, "/users/adamcomer/PycharmProjects/TensorflowSeq/Models/model.ckpt-4")

logloss = -1.0

loss = []

for j in range(0, 1000):
    for i in range(trainiterations):
        cutime = time.time() * 1000
        # data, outputs = sess.run([features, labels])
        outputs = ""
        data = raw_input("> ")

        data = np.reshape(np.array([data]), [1, 1])
        outputs = np.reshape(np.array([outputs]), [1, 1])

        databatch = []

        print(data)

        for line in data:
            print(line[0])
            data = re.split("\s", line[0])
            #print(data)
            tempdata = []
            for word in data:
                if word != '':
                    tempdata.append(dictionary[word])
            for p in range(seq_size - len(tempdata)):
                tempdata.append(dictionary["NULL"])
            tempdata.reverse()
            #print(tempdata)
            databatch.append(tempdata)

        labelsbatch = []
        tbatch = []

        for line in outputs:
            outputs = re.split("\s", line)
            outputs.insert(0, "GO")
            tempoutputs = []
            for word in outputs:
                if word != '':
                    tempoutputs.append(dictionary[word])
            for p in range(seq_size - len(tempoutputs)):
                tempoutputs.append(dictionary["NULL"])
            #print(tempoutputs)
            t = [tempoutputs[k + 1] for k in range(len(tempoutputs) - 1)]
            t = np.append(np.array(t), dictionary["NULL"])
            #print(t)
            labelsbatch.append(tempoutputs)
            tbatch.append(t)

        databatch = np.array(databatch)
        labelsbatch = np.array(labelsbatch)
        tbatch = np.array(tbatch)

        tempout = np.array(sess.run(rnn, feed_dict={x: databatch, y: labelsbatch, targets: tbatch}))
        print(tempout.shape)
        tempdata = []
        numtempdata = []
        for word in tempout:
            tempdata.append(rvsdictionary[np.argmax(word)])
            numtempdata.append(np.argmax(word))
        tempdata = [item for item in tempdata if item != 'NULL']
        print(tempdata)

        #print("Time: " + str((time.time() * 1000) - cutime) + " Batch: " + str(i) + " Iteration: " + str(j) + " Loss: " + str(logloss))

    print("Loss: " + str((total / (trainiterations * batch_size))))
    logloss = (total / (trainiterations * batch_size))
    saver.save(sess, "/Users/adamcomer/PycharmProjects/TensorflowSeq/Models/model.ckpt", global_step=j)
    total = 0

    loss.append(np.array([j, logloss]))
    np.savetxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/LogLoss.csv", np.array(loss), delimiter=",")

coord.request_stop()
coord.join(threads)
