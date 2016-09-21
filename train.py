# Adam Comer
# MIT Licence

import tensorflow as tf
import numpy as np
import time
import re
from itertools import izip
import argparse
import seq2seq_gpu

"""
    Function to train the model. This function will train a seq2seq model with 1 to 1 dialog pairs.
"""
def train_model(path, in_seq_size, out_seq_size, units, layers, trainiterations, batch_size, dout, dictionary, restore=False):

    # Placeholder variables to be feed into the model at run time
    x = tf.placeholder(tf.int32, shape=[None, None])
    y = tf.placeholder(tf.int32, shape=[None, None])
    targets = tf.placeholder(tf.int32, shape=[None, None])
    keep = tf.placeholder(tf.float32)

    # Dictionary reversed to lookup words from values
    rvsdictionary = dict(izip(dictionary.values(), dictionary.keys()))
    # Number of words in the dictionary
    dictsize = len(dictionary)

    # Files that the model is trained on
    filename_queue = tf.train.string_input_producer([str(path) + "dialogs.csv"])

    # Reads the files in the filename_queue
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Decodes the CSV to read the 1 pair string for the training data
    record_defaults = [[""], [""]]
    col1, col2 = tf.decode_csv(value, record_defaults=record_defaults, field_delim=",")

    # Constructs 2 tensors for the features(Input sentences) and labels(correct outputs)
    features = tf.pack(col1)
    labels = tf.pack(col2)

    # Shuffles the inputs
    features, labels = tf.train.shuffle_batch([features, labels], batch_size, capacity=10000, min_after_dequeue=5000, num_threads=4)

    # Arrays to hold the inputs and the correct outputs
    teminp = []
    temoutput = []
    temtarget = []

    # Makes the list of inputs for the rnn
    for o in range(in_seq_size):
        teminp.append(x[:, o])

    # Makes the list of inputs for the rnn
    for o in range(out_seq_size):
        temoutput.append(y[:, o])
        temtarget.append(targets[:, o])

    # Makes the temporary weights to train the model
    W1 = tf.placeholder(tf.float32, shape=[batch_size, out_seq_size])
    W1_0 = []
    for j in range(out_seq_size):
        W1_0.append(W1[:, j])

    # Makes the rnn cell(Gated Recurrent Unit(rnn cell alternative))
    cell1 = tf.nn.rnn_cell.GRUCell(units)
    # Adds dropout in the layers to make the model more robust
    drop = tf.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=keep, output_keep_prob=keep)
    # Makes multiple layers of the model
    cell = tf.nn.rnn_cell.MultiRNNCell([drop] * layers)

    # Number of samples for sampled softmax
    num_samples = 512

    # Makes the output projection layer by creating the variables the weights(w) and the bias(b)
    w = tf.get_variable("proj_w", [units, dictsize])
    w_t = tf.transpose(w)
    with tf.device("/cpu:0"):
        b = tf.get_variable("proj_b", [dictsize])
    # Output projection to take the rnn outputs and turn them into the word outputs
    output_projection = (w, b)

    # Sampling function to test the outputs and train the model
    def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, dictsize)

    # Declares the softmax loss function for training
    softmax_loss_function = sampled_loss

    # Seq2Seq model
    rnn, state = seq2seq_gpu.embedding_attention_seq2seq(teminp, temoutput, cell, dictsize, dictsize, 1000, output_projection=output_projection, feed_previous=False)

    rnnoutputs = [tf.matmul(word, w) + b for word in rnn]

    # Loss function to train the model
    logits = tf.nn.seq2seq.sequence_loss(rnn, temtarget, W1_0, softmax_loss_function=softmax_loss_function)
    tf.scalar_summary("Loss", logits)

    # Optimizer to change the weights in the model
    train = tf.train.AdagradOptimizer(0.1).minimize(logits)

    # Saves the model after training
    saver = tf.train.Saver()

    # GPU config files to control memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Initializes all the variables and creates all the Tensorflow objects above
    init_op = tf.initialize_all_variables()
    sess = tf.InteractiveSession(config=config)
    sess.run(init_op)

    # Takes data for training to be easily viewed
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(path + "graph", sess.graph)

    # Starts the treads for training
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print([v.name for v in tf.trainable_variables() if "embedding" in v.name])

    # If true will restore the model from a previous version
    if restore:
        print("Restoring Model")
        saver.restore(sess, str(path) + "model.ckpt")

    #print(sess.run(tf.get_default_graph().get_tensor_by_name("embedding_attention_seq2seq/embedding_attention_decoder/embedding:0")))

    # Training loop
    for i in range(trainiterations):
        # Gets the time to visualize training times
        cutime = time.time() * 1000

        # Gets the data from the dialogs file
        data, outputs = sess.run([features, labels])

        # Lists of the data to be trained
        databatch = []
        labelsbatch = []
        correctoutputsbatch = []

        # Loop to turn each word in the training sentences into integer arrays
        for line in data:
            # Splits the data by word
            data = re.split("\s", line)
            # Creates a list of the word integers
            tempdata = []
            # Fills the tempdata list with the word integers
            for word in data:
                if dictionary.get(word) is not None:
                    tempdata.append(dictionary[word])
                else:
                    tempdata.append(dictionary["UKN"])
            # Fills the rest of the empty spaces with null values
            for p in range(in_seq_size - len(tempdata)):
                tempdata.append(dictionary["NULL"])
            # Reverses the integers This has been show to make the model better
            tempdata.reverse()
            # Adds this sentence to the batch
            databatch.append(tempdata)

        # Loop to turn each word in the training sentences into integer arrays
        for line in outputs:
            # Splits the data by word
            outputs = re.split("\s", line)
            # Creates a list of the word integers
            outputs.insert(0, "GO")
            tempoutputs = []
            # Fills the tempoutputs list with the word integers
            for word in outputs:
                if dictionary.get(word) is not None:
                    tempoutputs.append(dictionary[word])
            # Fills the rest of the empty spaces with null values
            for p in range(out_seq_size - len(tempoutputs)):
                tempoutputs.append(dictionary["NULL"])
            # Makes the correct outputs to train the model on
            correctoutputs = [tempoutputs[k + 1] for k in range(len(tempoutputs) - 1)]
            correctoutputs = np.append(np.array(correctoutputs), dictionary["NULL"])
            # Adds the lists to the batches for training
            labelsbatch.append(tempoutputs)
            correctoutputsbatch.append(correctoutputs)

        # Makes the batches into arrays to be used by Tensorflow
        databatch = np.array(databatch)
        labelsbatch = np.array(labelsbatch)
        correctoutputsbatch = np.array(correctoutputsbatch)

        # Training action to change the weights of the model
        summery, _ = sess.run([merged, train], feed_dict={x: databatch, y: labelsbatch, targets: correctoutputsbatch, W1: np.ones([batch_size, out_seq_size], dtype=np.float32), keep: 0.5})
        #print(sess.run(tf.get_default_graph().get_tensor_by_name("embedding_attention_seq2seq/embedding_attention_decoder/embedding:0")))
        # Writes data to a file to be viewed
        writer.add_summary(summery, global_step=i)

        if dout:
            tempout = sess.run(rnnoutputs, feed_dict={x: databatch, y: labelsbatch, keep: 1.0})
            tempdata = np.split(np.array(tempout), batch_size, 1)

            data = []
            for sent in tempdata:
                temdata = []

                for word in sent:
                    temdata.append(rvsdictionary[np.argmax(word)])
                temdata = [item for item in temdata if item != 'NULL']
                data.append(temdata)
            print(data)
        print("Time: " + str((time.time() * 1000) - cutime) + " Iteration: " + str(i))

        if i % 10000 == 0 and i is not 0:
            saver.save(sess, str(path) + "model.ckpt", global_step=i)
            print("Model Saved")

    saver.save(sess, str(path) + "model.ckpt")

    coord.request_stop()
    coord.join(threads)

def clean(text):
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    text = text.lower()
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("[", " [ ")
    text = text.replace("]", " ] ")
    text = text.replace("{", " { ")
    text = text.replace("}", " } ")
    text = text.replace("0", " 0 ")
    text = text.replace("1", " 1 ")
    text = text.replace("2", " 2 ")
    text = text.replace("3", " 3 ")
    text = text.replace("4", " 4 ")
    text = text.replace("5", " 5 ")
    text = text.replace("6", " 6 ")
    text = text.replace("7", " 7 ")
    text = text.replace("8", " 8 ")
    text = text.replace("9", " 9 ")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("/", " / ")
    text = text.replace("$", " $ ")
    text = text.replace("%", " % ")
    text = text.replace(":", " : ")
    text = text.replace(";", " ; ")
    text = text.replace("' ", " ' ")
    text = text.replace(" '", " ' ")
    text = text.replace('"', " ")
    text = text.replace("-", " - ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("^", " ")
    text = text.replace("&", " & ")
    text = text.replace("#", " # ")
    text = text.replace("@", " ")
    text = text.replace("`", "'")
    text = text.replace("~", " ")
    text = text.split()

    return " ".join(text)


def create_dictionary(path, load):

    csv = np.loadtxt(str(path) + "dialogs.csv", delimiter=",", dtype="string", skiprows=0, comments="")

    tempdict = []
    dictionary = {"temp"}

    inmaxlen = 0
    outmaxlen = 0
    counter = 0
    for row in csv:
        for line in row:
            line = clean(line)
            words = re.split("\s", line)
            tempdict.extend(words)

            if counter == 0:
                if len(words) > inmaxlen:
                    inmaxlen = len(words)
            else:
                if len(words) > outmaxlen:
                    outmaxlen = len(words)
            counter += 1
        counter = 0

    if not load:
        tempdictionary = dictionary.union(tempdict)
        dictionary = ["NULL", "GO", "UKN"]

        # Creates a dictionary THIS CAN TAKE A FEW HOURS!
        # for item in tempdictionary:
        #     if tempdict.count(item) >= 50: # Change this number to raise or lower the word frequency minimum counter
        #         dictionary.append(item)

        # Uncomment this line and comment the for loop to use all the words in the training data
        dictionary = dictionary + list(tempdictionary)

        np.savetxt(str(path) + "dictionary.csv", dictionary, fmt="%s", delimiter=",")

    dictionary = np.loadtxt(str(path) + "dictionary.csv", delimiter="^", dtype="string", skiprows=0, comments="")

    dictsize = len(dictionary)

    count = []

    for i in range(dictsize):
        count.append(i)

    dictionary = dict(izip(list(dictionary), count))

    print("Number of words in dictionary: " + str(len(dictionary)))
    print("Max Input Length: " + str(inmaxlen))
    print("Max Output Length: " + str(outmaxlen))
    print("Number of lines: " + str(csv.shape[0]))

    return int(inmaxlen), int(outmaxlen), dictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dialog_path", type=str, help="path to the dialog csv ex: \"/user/dialog_folder/\"")
    parser.add_argument("units", type=int, help="number of units in a GRU layer")
    parser.add_argument("layers", type=int, help="number of GRU layers")
    parser.add_argument("train_iterations", type=int, help="number of training iterations")
    parser.add_argument("batch_size", type=int, help="number of units in a batch")
    parser.add_argument("--restore", help="if you want to restore from a old model", action="store_true")
    parser.add_argument("--display_out", help="if you want to see the outputs from training", action="store_true")
    parser.add_argument("--load_dictionary", help="if you have a dictionary from a past training iteration", action="store_true")
    args = parser.parse_args()

    inlength, outlength, dictionary = create_dictionary(args.dialog_path, args.load_dictionary)

    if args.restore:
        train_model(args.dialog_path, inlength + 1, outlength + 1, args.units, args.layers, args.train_iterations, args.batch_size, args.display_out, dictionary, restore=args.restore)
    else:
        train_model(args.dialog_path, inlength + 1, outlength + 1, args.units, args.layers, args.train_iterations, args.batch_size, args.display_out, dictionary)