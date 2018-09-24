import numpy as np
import preprocess_data
import tensorflow as tf

def doWork(archive_path, nb_iterations=10000, embedding_size=100, lstm_size=100, max_sentence_length=None, logdir=None, save_every=None, save_to=None):
    # Constants
    batch_size = 100
    
    data, label, length, test_data, test_label, test_length, vocab_size = preprocess_data.doCache('/tmp/preprocessed_data_'+ str(max_sentence_length), lambda: preprocess_data.load_transformed_data(archive_path, max_sentence_length))
    #print("Reviews data:")
    #tmplength = np.concatenate((length, test_length))
    #print("Avg: {}".format(sum(tmplength)/len(tmplength)))
    #print("Min: {}".format(min(tmplength)))
    #print("Max: {}".format(max(tmplength)))
    #print("Preprocessing done!")
    #exit()
    # We're going to need this
    if not max_sentence_length:
        max_sentence_length = max(np.concatenate((length, test_length)))
    
    sentence_length = data.shape[1]
    nb_classes = 2 # Positive/Negative
    X = tf.placeholder(tf.int32, [None, sentence_length], name="input_data") # Input data
    X_length = tf.placeholder(tf.int32, [None], name="input_data_length") # Length of each input
    Y = tf.placeholder(tf.uint8, [None], name="input_labels") # Input labels
    YY = tf.one_hot(Y, nb_classes)
    #dropout_pkeep = tf.placeholder(tf.float32, name="dropout_prob") # Dropout probability
    
    # Embedd the data
    with tf.name_scope('embedding'):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        #embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # Define the network
    with tf.name_scope("lstm"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
        output, state = tf.nn.dynamic_rnn(lstm_cell, embedded_chars, sequence_length=X_length, swap_memory=True, dtype=tf.float32) # TODO: Add sequence_length, to avoid working on padding.
        # Now collect the right frame
        # The last one, if sequence_length is not provided - - -
        #output = tf.transpose(output, [1,0,2])
        #last = tf.gather(output, int(output.get_shape()[0]-1))
        # - - -
        # Otherwise, the last relevant frame
        lstm_current_batch_size = tf.shape(output)[0]
        lstm_current_index = tf.range(0, lstm_current_batch_size) * max_sentence_length + (X_length-1)
        last = tf.gather(tf.reshape(output, [-1, lstm_size]), lstm_current_index)
        # - - -
    
    # The ouput layer (classification)!
    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([lstm_size, nb_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[nb_classes]))
        scores = tf.nn.xw_plus_b(last, W, b)

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, YY))
        summary_loss = tf.summary.scalar("loss", loss)

    # Calculate Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.argmax(scores, 1), tf.argmax(YY, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        summary_accuracy = tf.summary.scalar("accuracy", accuracy)
    
    # Prepare computation
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

    # Prepare the session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Enable logging
    merged = tf.summary.merge([summary_accuracy, summary_loss])
    if logdir:
        from os.path import join
        subfolder = str(nb_iterations) + "-" + str(lstm_size) + "-" + str(embedding_size) + "-" + str(max_sentence_length)
        train_writer = tf.summary.FileWriter(join(logdir, "train", subfolder), graph=sess.graph)
        test_writer = tf.summary.FileWriter(join(logdir,"test", subfolder), graph=sess.graph)
    
    # Prepare to save the graph
    saver = tf.train.Saver()

    # Train
    try:
        for iteration in range(nb_iterations):
            if save_every and iteration%save_every==0:
                saver.save(sess, "/tmp/tensorflow-save/"+str(iteration) + "-" + str(lstm_size) + "-" + str(embedding_size) + "-" + str(max_sentence_length) + ".ckpt")
            # Test one minibatch
            if(iteration % 100 == 0):
                summary, acc, step = sess.run([merged, accuracy, global_step], feed_dict={X:test_data[:batch_size], Y:test_label[:batch_size], X_length:test_length[:batch_size]})
                #if logdir:
                #    test_writer.add_summary(summary, step)
            # Test the whole set
            if(iteration % 100 == 0):
                train_acc = []
                a = []
                l = []
                for i in range(int(len(test_data) / batch_size)):
                    aa, ll = sess.run([accuracy, loss], feed_dict={X:test_data[i * batch_size:(i + 1) * batch_size], Y:test_label[i * batch_size:(i + 1) * batch_size], X_length:test_length[i * batch_size:(i + 1) * batch_size]})
                    a += [aa]
                    l += [ll]
                    print("\r{}\t{}%".format(iteration, i * 100 / len(test_data) * 100), end="  ")
                a = sum(a)/len(a)
                l = sum(l)/len(l)
                if logdir:
                    test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/accuracy", simple_value=a)]),step)
                    test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/loss", simple_value=l)]),step)
            print("\r{}\t{}/{}".format(iteration, a, l), end="            ")
            stepX = data[iteration * batch_size % data.shape[0]:iteration * batch_size % data.shape[0] + batch_size]
            stepY = label[iteration * batch_size % len(label):iteration * batch_size % len(label) + batch_size]
            stepXlength = length[iteration * batch_size % len(length):iteration * batch_size % len(length) + batch_size]
            summary,_, step = sess.run([merged, train_step, global_step], feed_dict={X:stepX, Y:stepY, X_length:stepXlength})
            if logdir:
                train_writer.add_summary(summary, step)
    except KeyboardInterrupt:
        print("Training interrupted.")
    print("")
    
    # Test
    print("Testing...")
    a = []
    l = []
    for i in range(int(len(test_data) / batch_size)):
        aa, ll = sess.run([accuracy, loss], feed_dict={X:test_data[i * batch_size:(i + 1) * batch_size], Y:test_label[i * batch_size:(i + 1) * batch_size], X_length:test_length[i * batch_size:(i + 1) * batch_size]})
        a += [aa]
        l += [ll]
        print("\r{}%".format(i * 100 / len(test_data) * 100), end="  ")
    print()
    print("Final accuracy/loss: {}/{}".format(sum(a) / len(a), sum(l) / len(l)))
    
    if save_to:
        from os.path import join
        path = saver.save(sess, join(save_to, str(iteration) + "-" + str(lstm_size) + "-" + str(embedding_size) + "-" + str(max_sentence_length) + ".ckpt"))
        print("Graph saved to '{}'.".format(path))
    
if __name__ == "__main__":
    import argparse
    # CLI arguments parsing
    parser = argparse.ArgumentParser(description="Do sentiment analysis on the Stanford Large Movie review dataset using a convolutional network.")
    parser.add_argument('archive_path', type=str, help='The tar.gz archive of the dataset.')
    parser.add_argument('--iterations', type=int, default=10000, help='The amount of iterations to perform.')
    parser.add_argument('--embedding', type=int, default=100, help='The embedding size to use.')
    parser.add_argument('--lstm-size', type=int, default=100, help='The amount of neurones to use in the RNN unit.')
    parser.add_argument('--sentence-length', type=int, default=None, help='The maximum sentence length.')
    parser.add_argument('--logdir', type=str, default=None, help='Root log directory.')
    parser.add_argument('--save-every', type=int, default=None, help='Save the network every given iterations.')
    parser.add_argument('--save', type=str, default=None, help='Save the network at the end.')
    options = parser.parse_args()
	
    doWork(options.archive_path, options.iterations, options.embedding, options.lstm_size, options.sentence_length, logdir=options.logdir, save_every=options.save_every, save_to=options.save)
