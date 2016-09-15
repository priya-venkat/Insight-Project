#!/usr/bin/env python


import os
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class cleanquest(object):
    """cleanquest is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review, "lxml").get_text()
        #
        #  Remove q: 
        review_text = re.sub('q:', '', review_text)
        #  Remove a: 
        review_text = re.sub('a:', '', review_text)
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=True ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( cleanquest.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
        
def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( cleanquest.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews
    

 # Read data from files
train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'new_quest_id_review.tsv'), header = 0, delimiter="\t", quoting=3 )
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read" 

print str(train)
#print str(test)

print train["review"][1]
#print test["review"][1]

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# ****** Split the labeled and unlabeled training sets into clean sentences
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += cleanquest.review_to_sentences(review, tokenizer)

print sentences[1:3]

# Set values for various parameters
num_features = 30   # Word vector dimensionality
min_word_count = 1   # Minimum word count   # default min is 5
num_workers = 4       # Number of threads to run in parallel
context = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
### model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model = Word2Vec(sentences, workers=num_workers, \
			size=num_features, min_count = min_word_count, \
			window = context, sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "just3questions_ver1"
model.save(model_name)

# model.doesnt_match("man woman child kitchen".split())
print model.most_similar("britain")
print model.most_similar("ford")


# how about printing summation of word vectors for one question, 
# then generalize to print for all questions

# sent = [u'calories', u'handful', u'strawberries']
# sent1 = [u'many', u'calories', u'cup', u'strawberrys']
# model = Word2Vec.load('just3questions')

senvec = np.array([])
#senvec = np.array((30,))
#data = np.loadtxt('foo.csv')
for sent in sentences:
	vec = np.zeros((30,))
	for word in sent: 
		vec += model[word] 
		print vec.shape
	#senvec = np.hstack([senvec, vec])
	senvec = np.append(senvec.T, vec.T, axis = 0)
print senvec[1]
print senvec[2]
np.savetxt("senvec.csv", senvec, delimiter = "\t")


#sentence[1]
	

# new question
# newq = [u'calories', u'cup', u'strawberry']
# vect = np.zeros((30,))
# for word in newq: 
# 	vect += model[word] 
# print vect
# 
# vect1 = np.array([-0.3581191 ,  0.64951618,  0.30497545,  0.18290451, -0.16889866,
# 0.21647574, -0.534729  ,  0.07829019,  0.18007717, -0.09345566,
# -0.18942919, -0.02728072, -0.13680503, -0.08807184,  0.45360228,
# 0.02772516, -0.16706966,  0.34171431,  0.25425042, -0.14813522,
# 0.10110275, -0.41969806,  0.8039929 ,  0.34847383, -0.10586351,
# -0.38279241,  0.4168102 , -0.26910328,  0.14293349,  0.03805288])

# cosine similarity

# prod = np.





