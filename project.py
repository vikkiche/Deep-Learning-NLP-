import numpy as np
import preprocess_data
import tensorflow as tf
import matplotlib.pyplot as plt

# Preprocess the data: extract labels and split by words
print("Loading data...")
#label, data = preprocess_data.load_data_and_label("C:\\Users\\alexi\\Downloads\\aclImdb\\train\\pos", "C:\\Users\\alexi\\Downloads\\aclImdb\\train\\neg")
#test_label, test_data = preprocess_data.load_data_and_label("C:\\Users\\alexi\\Downloads\\aclImdb\\test\\pos", "C:\\Users\\alexi\\Downloads\\aclImdb\\test\\neg")
#label, data = preprocess_data.from_pickle("C:\\Users\\alexi\\Downloads\\aclImdb\\train\\pickle")
#test_label, test_data = preprocess_data.from_pickle("C:\\Users\\alexi\\Downloads\\aclImdb\\test\\pickle")
label, data = preprocess_data.from_pickle("/data/aclImdb/train/pickle")
test_label, test_data = preprocess_data.from_pickle("/data/aclImdb/test/pickle")
print("Successfuly loaded {:d} training reviews and {:d} test reviews.".format(len(data), len(test_data)))
# Shuffle the reviews
p = np.random.permutation(len(label))
label = np.array(label)[p]
data = np.array(data)[p]
p = np.random.permutation(len(test_label))
test_label = np.array(test_label)[p]
test_data = np.array(test_data)[p]
print("Converting words...")
# Learn the vocabulary
max_length = max([len(x.split(" ")) for x in np.concatenate((data, test_data))])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)
data = np.array(list(vocab_processor.fit_transform(data)))
test_data = np.array(list(vocab_processor.fit_transform(test_data)))
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("Preprocessing done!")

sentence_length = data.shape[1]
nb_classes = 2 # Positive/Negative
X = tf.placeholder(tf.int32, [None, sentence_length], name="X") # Input data
Y = tf.placeholder(tf.uint8, [None], name="Y") # Input labels
YY = tf.one_hot(Y, nb_classes)
dropout_pkeep = tf.placeholder(tf.float32, name="dropout_prob") # Dropout probability
embedding_size = 100 # How many closely related words do we find?

# Embedd the data
with tf.name_scope('embedding'):
    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, X)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create the convolutional layers
pooled_outputs = []
filter_sizes = [3, 4, 5]
num_filters = 4 # The amount of features to extract
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        W = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
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
num_filters_total = num_filters * len(filter_sizes) # Total amount of features extracted
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

# Prepare computation
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

# Do the job!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

nb_iterations = 10000
batch_size = 100
train_acc = []
import itertools
try:
    #for iteration in range(nb_iterations):
    for iteration in itertools.count():
        if(iteration % 100 == 0):
            acc = sess.run(accuracy, feed_dict={X:test_data[:batch_size], Y:test_label[:batch_size], dropout_pkeep:1})
            train_acc += [acc]
        print("\r" + str(iteration) + "\t" + str(acc), end="            ")
        stepX = data[iteration * batch_size % data.shape[0]:iteration * batch_size % data.shape[0] + batch_size]
        stepY = label[iteration * batch_size % len(label):iteration * batch_size % len(label) + batch_size]
        sess.run(train_step, feed_dict={X:stepX, Y:stepY, dropout_pkeep:0.5})
except KeyboardInterrupt:
    print("Interrupted training.")
print("")
print("Testing...")
a = []
l = []
for i in range(int(len(test_data) / batch_size)):
    aa, ll = sess.run([accuracy, loss], feed_dict={X:test_data[i * batch_size:(i + 1) * batch_size], Y:test_label[i * batch_size:(i + 1) * batch_size], dropout_pkeep:1})
    a += [aa]
    l += [ll]
    print("\r{}%".format(i * 100 / len(test_data) * 100), end="  ")
print()
print("Final accuracy/loss: {}/{}".format(sum(a) / len(a), sum(l) / len(l)))

# Plot!
plt.plot(train_acc)
plt.title("Accuracy")
plt.grid(True)
plt.show()
