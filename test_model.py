# Copyright 2016 Adam Comer. All rights reserved. For public use only.

import argparse
import re
from itertools import izip
import numpy as np
import tensorflow as tf
from model import create_dictionary


def test_model(path, seq_size, units, layers, dictionary):
    x = tf.placeholder(tf.int32, shape=[None, None])
    y = tf.placeholder(tf.int32, shape=[None, None])
    targets = tf.placeholder(tf.int32, shape=[None, None])

    dictsize = len(dictionary)
    rvsdictionary = dict(izip(dictionary.values(), dictionary.keys()))

    teminp = []
    temoutput = []
    temtarget = []

    for o in range(seq_size):
        teminp.append(x[:, o])
        temoutput.append(y[:, o])
        temtarget.append(targets[:, o])

    cell1 = tf.nn.rnn_cell.GRUCell(units)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1] * layers)

    rnn, state = tf.nn.seq2seq.embedding_attention_seq2seq(teminp, temoutput, cell, dictsize, dictsize, 100, feed_previous=True)

    saver = tf.train.Saver()

    init_op = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, str(path) + "model.ckpt")

    while True:
        outputs = ""
        data = raw_input("> ")

        data = np.reshape(np.array([data]), [1, 1])
        outputs = np.reshape(np.array([outputs]), [1, 1])

        databatch = []

        for line in data:
            data = re.split("\s", line[0])
            tempdata = []
            for word in data:
                if word != '':
                    if dictionary.get(word) is not None:
                        tempdata.append(dictionary[word])
                    else:
                        tempdata.append(dictionary["NULL"])
            for p in range(seq_size - len(tempdata)):
                tempdata.append(dictionary["NULL"])
            tempdata.reverse()
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
            t = [tempoutputs[k + 1] for k in range(len(tempoutputs) - 1)]
            t = np.append(np.array(t), dictionary["NULL"])
            labelsbatch.append(tempoutputs)
            tbatch.append(t)

        databatch = np.array(databatch)
        labelsbatch = np.array(labelsbatch)
        tbatch = np.array(tbatch)

        tempout = np.array(sess.run(rnn, feed_dict={x: databatch, y: labelsbatch, targets: tbatch}))
        tempdata = []
        numtempdata = []
        for word in tempout:
            tempdata.append(rvsdictionary[np.argmax(word)])
            numtempdata.append(np.argmax(word))
        tempdata = [item for item in tempdata if item != 'NULL']
        print(' '.join(tempdata))

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dialog_path", type=str, help="path to the dialog csv ex: \"/user/dialog_folder/\"")
    parser.add_argument("units", type=int, help="number of units in a GRU layer")
    parser.add_argument("layers", type=int, help="number of GRU layers")
    args = parser.parse_args()

    maxlength, dictionary = create_dictionary(args.dialog_path)

    test_model(args.dialog_path, maxlength, args.units, args.layers, dictionary)
