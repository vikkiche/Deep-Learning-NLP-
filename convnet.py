import numpy as np
import preprocess_data
import tensorflow as tf
import matplotlib.pyplot as plt

# embedding_size is the amount of closely related words to use
# filter_size 
def doWork(archive_path, nb_iterations, embedding_size=100, filter_sizes = [3, 4, 5], num_features=4, max_sentence_length=None, logdir=None, do_plot=False):
    """
    Train a model that does some binary sentiment analysis on the Big Movie Review dataset.
    
    nb_iterations -- The amount of iterations to perform. Note that the batch size if fixed to 100.
    embedding_size -- The amount of closely related words to load for each word.
    filter_sizes -- The sizes of the filters to use. See those as sliding windows over the sentence, a filter of size 3 will observe three words at once.
    num_features -- The amount of features to extract in the convolutional layers.
    max_sentence_length -- The maximum length of a sentence. If it is less than the actual length a a sentence, it will be truncated. If it is more than it's actual size, it is zero-padded. Don't specify to use the longuest sentence as reference.
    logdir -- The logging directory to use with tensorboad.
    do_plot -- Should it print a plot of the accuracy at the end.
    """

    data, label, length, test_data, test_label, test_length, vocab_size = preprocess_data.doCache('/tmp/preprocessed_data_'+ str(max_sentence_length), lambda: preprocess_data.load_transformed_data(archive_path, max_sentence_length))
    #print("Sentence data:")
    #tmplength = np.concatenate((length, test_length))
    #print("Avg: {}".format(sum(tmplength)/len(tmplength)))
    #print("Min: {}".format(min(tmplength)))
    #print("Max: {}".format(max(tmplength)))
    #print("Preprocessing done!")
    #exit()
    # We're going to need this
    if not max_sentence_length:
        max_sentence_length = max(np.concatenate((length, test_length)))
    print("Preprocessing done!")

    sentence_length = data.shape[1]
    nb_classes = 2 # Positive/Negative
    X = tf.placeholder(tf.int32, [None, sentence_length], name="X") # Input data
    Y = tf.placeholder(tf.uint8, [None], name="Y") # Input labels
    YY = tf.one_hot(Y, nb_classes)
    dropout_pkeep = tf.placeholder(tf.float32, name="dropout_prob") # Dropout probability
    
    # Embedd the data
    with tf.name_scope('embedding'):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # Create the convolutional layers
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_features], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_features]))
            conv = tf.nn.conv2d(embedded_chars_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, sentence_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_features * len(filter_sizes) # Total amount of features extracted
    h_pool = tf.concat(3, pooled_outputs) # Combine the outputs of the convolutions
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # Flatten it

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_pkeep)

    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([num_filters_total, nb_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[nb_classes]))
        scores = tf.nn.xw_plus_b(h_drop, W, b)

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, YY))

    # Calculate Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.argmax(scores, 1), tf.argmax(YY, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    # Prepare computation
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

    # Prepare the session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Enable logging
    merged = tf.summary.merge_all()
    if logdir:
        from os.path import join
        subfolder = str(nb_iterations) + "-"
        for i in range(len(filter_sizes)):
            subfolder += str(filter_sizes[i])
            if i!=len(filter_sizes)-1:
                subfolder+="_"
        subfolder +="-"+str(num_features)+"-"+str(embedding_size) +"-"+ str(max_sentence_length)
        train_writer = tf.summary.FileWriter(join(logdir, "train", subfolder), graph=sess.graph)
        test_writer = tf.summary.FileWriter(join(logdir,"test", subfolder), graph=sess.graph)

    # Do the job!
    batch_size = 100
    train_acc = []
    try:
        for iteration in range(nb_iterations):
        #import itertools
        #for iteration in itertools.count():
            if(iteration % 100 == 0):
                summary, acc, step = sess.run([merged, accuracy, global_step], feed_dict={X:test_data[:batch_size], Y:test_label[:batch_size], dropout_pkeep:1})
                train_acc += [acc]
                if logdir:
                    test_writer.add_summary(summary, step)
            print("\r" + str(iteration) + "\t" + str(acc), end="            ")
            stepX = data[iteration * batch_size % data.shape[0]:iteration * batch_size % data.shape[0] + batch_size]
            stepY = label[iteration * batch_size % len(label):iteration * batch_size % len(label) + batch_size]
            summary,_, step = sess.run([merged, train_step, global_step], feed_dict={X:stepX, Y:stepY, dropout_pkeep:0.5})
            if logdir:
                train_writer.add_summary(summary, step)
    except KeyboardInterrupt:
        print("Training interrupted.")
    print("")
    print("Testing...")
    a = []
    l = []
    s = []
    for i in range(int(len(test_data) / batch_size)):
        aa, ll = sess.run([accuracy, loss], feed_dict={X:test_data[i * batch_size:(i + 1) * batch_size], Y:test_label[i * batch_size:(i + 1) * batch_size], dropout_pkeep:1})
        a += [aa]
        l += [ll]
        print("\r{}%".format(i * 100 / len(test_data) * 100), end="  ")
    print()
    print("Final accuracy/loss: {}/{}".format(sum(a) / len(a), sum(l) / len(l)))

    # Plot?
    if do_plot:
        plt.plot(train_acc)
        plt.title("Accuracy")
        plt.grid(True)
        plt.show()
        
    sess.close()

if __name__ == "__main__":
    import argparse
    # CLI arguments parsing
    parser = argparse.ArgumentParser(description="Do sentiment analysis on the Stanford Large Movie review dataset using a convolutional network.")
    parser.add_argument('archive_path', type=str, help='The tar.gz archive of the dataset.')
    parser.add_argument('--iterations', type=int, default=10000, help='The amount of iterations to perform.')
    parser.add_argument('--embedding', type=int, default=100, help='The embedding size to use. This correspond to the amount of closely related words that we should use.')
    parser.add_argument('--filters', type=lambda s:[int(item) for item in s.split(',')], default="3,4,5", help='Sizes of the filters.')
    parser.add_argument('--features', type=int, default=4, help='The amount of features to extract in the convolutional layer.')
    parser.add_argument('--sentence-length', type=int, default=None, help='.')
    parser.add_argument('--logdir', type=str, default=None, help='Root log directory.')
    parser.add_argument('--show-graph', action='store_true', default=False, help='.')
    options = parser.parse_args()
	
    doWork(options.archive_path, options.iterations, options.embedding, options.filters, options.features, options.sentence_length, options.logdir, options.show_graph)
        
