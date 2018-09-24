import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_label_targz(filepath):
    """
    Load the data and labels from the Large Movie Review dataset, directly from 
    the archive.
    """
    import gzip, tarfile, io
    unzipped = gzip.GzipFile(filepath)
    file_like = io.BytesIO(unzipped.read())
    tar = tarfile.open(fileobj=file_like)
    tar.getmembers()
    
    data = []
    labels = []
    test_data = []
    test_labels = []
    for member in tar.getmembers():
        match = re.search("/(test|train)/(pos|neg)/[0-9]*_([0-9]*)\.txt", member.name)
        if match:
            try:
                tmp_data = clean_str(tar.extractfile(member).read().decode("utf-8"))
                if match.group(1)=="test":
                    test_data += [tmp_data]
                    test_labels += [1 if int(match.group(3))>5 else 0]
                else:
                    data += [tmp_data]
                    labels += [1 if int(match.group(3))>5 else 0]
            except UnicodeDecodeError:
                print("Skipping file '" + file + "': Unicode error.")
    return data, labels, test_data, test_labels

def load_data_and_label(*folders):
    """
    Load the data and labels from the Large Movie Review dataset, from the 
    extracted files. The extraction directly from the archive is recommended 
    since this function incurs a lot of potentially non-contiguous file access.
    """
    from os import listdir
    from os.path import isfile, join
    data = []
    labels = []
    for folder in folders:
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for file in files:
            #print(file)
            try:
                data += [clean_str(open(join(folder, file), 'r').read())]
                labels += [1 if int(re.match("[0-9]*_([0-9]*).txt", file).group(1)) > 5 else 0]
            except UnicodeDecodeError:
                print("Skipping file '" + file + "': Unicode error.")
    return labels, data

def to_pickle(dest, data):
    """
    Save the given data to the given file.
    """
    import _pickle as pickle
    with open(dest, "wb") as file:
        pickle.dump(data, file)

def from_pickle(file):
    """
    Load the data from the given pickle file.
    """
    import _pickle as pickle
    with open(file, "rb") as file:
        return pickle.load(file)

def doCache(filename, function):
    """
    On the first run, compute the value of the given function. Subsequent runs 
    will load the cached data from the generated file.
    """
    from os.path import isfile
    if isfile(filename):
        print("{} is already cached, loading from file...".format(filename))
        return from_pickle(filename)
    tmp = function()
    to_pickle(filename, tmp)
    return tmp

import numpy as np
np.random.seed(0)

def loadShuffled(archive_path):
    """
    Load the data from the dataset and shuffle it.
    """
    print("Loading data...")
    #label, data = doCache("/data/aclImdb/train/pickle", lambda: load_data_and_label("/data/aclImdb/train/pos", "/data/aclImdb/train/neg"))
    #test_label, test_data = doCache("/data/aclImdb/test/pickle", lambda: load_data_and_label("/data/aclImdb/test/pos", "/data/aclImdb/test/neg"))
    data, label, test_data, test_label = load_data_and_label_targz(archive_path)
    print("Successfuly loaded {:d} training reviews and {:d} test reviews.".format(len(data), len(test_data)))
    # Shuffle the reviews
    p = np.random.permutation(len(label))
    label = np.array(label)[p]
    data = np.array(data)[p]
    p = np.random.permutation(len(test_label))
    test_label = np.array(test_label)[p]
    test_data = np.array(test_data)[p]
    return label, data, test_label, test_data

def load_transformed_data(archive_path, max_sentence_length=None):
    """
    Load the data using the above functions, then convert the words using a 
    VocabularyProcessor.
    """
    import tensorflow as tf
    # Load the data (suffled)
    label, data, test_label, test_data = loadShuffled(archive_path)
    print("Converting words...")
    # Learn the vocabulary
    length = [len(x.split(" ")) for x in data]
    test_length = [len(x.split(" ")) for x in test_data]
    if not max_sentence_length:
        max_sentence_length = max(np.concatenate((length, test_length)))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sentence_length)
    data = np.array(list(vocab_processor.fit_transform(data)))
    test_data = np.array(list(vocab_processor.fit_transform(test_data)))
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    return data, label, length, test_data, test_label, test_length, vocab_size

