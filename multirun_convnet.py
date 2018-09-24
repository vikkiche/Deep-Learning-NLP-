from os.path import isdir, join
import convnet

if __name__ == "__main__":
    logdir="/data/tensorflow"
    nb_iterations=10000
    for embedding_size in [10, 50, 100, 200, 500]:
        for filter_sizes in [[3,4,5], [3,4,5,6,7]]:
            for num_features in [4,8,16]:
                for max_sentence_length in [100, 1000, 2633]:
                    print("Starting with " + str(nb_iterations) + " and " + str(filter_sizes) + " filters with " + str(num_features) + " features and a maximum sentence length of " + str(max_sentence_length) + ", with an embedding of "+str(embedding_size) + "...")
                    subfolder = str(nb_iterations) + "-"
                    for i in range(len(filter_sizes)):
                        subfolder += str(filter_sizes[i])
                        if i!=len(filter_sizes)-1:
                            subfolder+="_"
                    subfolder +="-"+str(num_features)+"-"+str(embedding_size) +"-"+ str(max_sentence_length)
                    subfolder = join(logdir, "train", subfolder)
                    if not isdir(subfolder):
                        convnet.doWork(nb_iterations, embedding_size=embedding_size, filter_sizes=filter_sizes, num_features=num_features, max_sentence_length=max_sentence_length, logdir=logdir, do_plot=False)
                        exit()
                    else:
                        print("'"+subfolder + "' already exists, skipping...")
