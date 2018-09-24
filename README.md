# Deep Learning Project
In this document, I describe my results and interpretations of the comparison of two neural networks made to analyse sentiment on the Stanford Large Movie Review dataset.
Data preprocessing
The dataset is composed of two subsets: one training set and one testing set. Each of those sets contains 25000 reviews collected from IMDB which we first split into words, then transform using the word2vec method, provided by tensorflow through the VocabularyPreprocessor class. This compute the cosine distance between each word and needs to be trained, so I will feed all the reviews to it. Note that I could use a pre-trained dictionary, which could include way more words, but I don’t think that it would be relevant in this situation, since I have using a determined data set. This pre-training phase gives us some information on the data-set:

Average sentence length
254.7
Minimum sentence length
8
Maximum sentence length
2633
Vocabulary size
80862

Each word is converted to a tensor as wide as I wish, called the embedding size. This size allows us to express how many different features I am going to learn for each unique word. Also, I must choose how I’m going to consider each review: since using convolutional layers which have fixed size, need to know beforehand the length of the sentences. Then need to pick the maximum length that we want to consider. All the reviews will then be either zero-padded of truncated to match the given maximum length.
The first implementation uses a convolutional neural network. In this network, I use a single convolutional layer with several different convolution operations: each unit uses a different size, which allow them to consider different amounts of words.
In this network, I could tune the number of convolutional networks and their sizes, the number of features to extract in each of them, the embedding size and the length of the reviews.
By tuning those hyperparameters, ended up with some interesting observations. I noticed that going above 8 features for the convolutional layers is useless, as it increases the computational time but has a neglectable effect on the accuracy compared to 8 features. Using less features, on the other hand, gives significantly lower accuracy and computation times. 
I tested our model with embedding of 10, 50 and 100. The highest value gave the best results most of the times. I believe that could use a value slightly lower than a hundred and still get good results, or go above this to try to get slightly better results.
As for the maximum sentence length, the results were good when using a value of a thousand, and using maximum length of 2633 (which is the length of the longest review in the model) gave similar results with a computation time more than two times longer.
However, we think that the necessity of specifying a maximum sentence length and using padding to attain it might cause a big loss of information. This problem does not exist in the case of LSTM networks.
Recurrent Neural Network
This network was made using a simple LSTM cell provided by tensorflow. The available hyperparameters were the number of neurones in the lstm cell and once again the embedding size. The maximum sentence length could also be specified but we neglected this one for this network since it does not make computations on the padded part of the reviews (I still need to pad it though, since a tensor need each row to be the same size).
This method is optional, but highly desired since the LSTM cell won’t obtain anything significant when working on the padding, and it will give awful results while greatly increasing the computation time.
However, I could observe that the RNN network allows us to obtain results with above 85% of accuracy, which is 5% better than the best CNN model that we found. Also, I observed that it required more memory to run with interesting parameters, which reduced our possibilities regarding experimentation without using a cluster, and more specifically prevented us from testing this network with more than a hundred LSTM cells.
