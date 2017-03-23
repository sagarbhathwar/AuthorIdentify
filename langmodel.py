from collections import Counter
from functools import reduce
from copy import deepcopy

import numpy as np
import nltk

class LangaugeModel:

    def __init__(self, bookname):
        self.bookname = bookname
        self.author = bookname.split('-')[0]
        
        self.unigrams = list(nltk.corpus.gutenberg.words(bookname))
        self.bigrams = list(nltk.bigrams(self.unigrams))
        self.trigrams = list(nltk.trigrams(self.unigrams))

        self.unigrams_c = Counter(self.unigrams)
        self.bigrams_c = Counter(self.bigrams)
        self.trigrams_c = Counter(self.trigrams)

        self.unigrams_p = {
            w: self.unigrams_c[w] / len(self.unigrams) for w in self.unigrams}
        self.bigrams_p = {
            w: self.bigrams_c[w] / self.unigrams_c[w[0]] for w in self.bigrams}
        self.trigrams_p = {
            w: self.trigrams_c[w] / self.bigrams_c[w[0:2]] for w in self.trigrams}

        self.smoothed_unigrams = deepcopy(self.unigrams)
        self.smoothed_bigrams = deepcopy(self.bigrams)
        self.smoothed_trigrams = deepcopy(self.trigrams)

        self.smoothed_unigrams_c = {}
        self.smoothed_bigrams_c = {}
        self.smoothed_trigrams_c = {}

        self.smoothed_unigrams_p = {}
        self.smoothed_bigrams_p = {}
        self.smoothed_trigrams_p = {}

    def smooth(self, unigrams, bigrams, trigrams):
        """
        Add one smoothing
        Each unigram, bigram and trigram contain all those ever possible in the 
        three books. So, updating each of them with 'everything' makes sure that ones which
        are already present is incremented by one while the rest are initialized to one
        Also the smoothed probabilities is calculated in a way similar to unsmoothed models 
        """
        self.smoothed_unigrams += unigrams
        self.smoothed_bigrams += bigrams
        self.smoothed_trigrams += trigrams
        
        self.smoothed_unigrams_c = Counter(self.smoothed_unigrams)
        self.smoothed_bigrams_c = Counter(self.smoothed_bigrams)
        self.smoothed_trigrams_c = Counter(self.smoothed_trigrams)

        
        n = len(self.smoothed_unigrams)
        self.smoothed_unigrams_p = {
            w: self.smoothed_unigrams_c[w] / n for w in self.smoothed_unigrams}
        self.smoothed_bigrams_p = {
            w: self.smoothed_bigrams_c[w] / self.smoothed_unigrams_c[w[0]] for w in self.smoothed_bigrams}
        self.smoothed_trigrams_p = {
            w: self.smoothed_trigrams_c[w] / self.smoothed_bigrams_c[w[0:2]] for w in self.smoothed_trigrams}

    def printSmooth(self):
        print(self.smoothed_unigrams_p)
        print(self.smoothed_bigrams_p)
        print(self.smoothed_trigrams_p)

    def calculate_cross_entropy_values(self):
        """
        returns the cross entorpy computer for unigram, bigram
        and trigram models w.r.t each other for this language model
        """
        u_ce = (-1 / len(self.unigrams)) * np.sum(np.log([self.unigrams_p[w] for w in self.unigrams]))
        b_ce = (-1 / len(self.bigrams)) * np.sum(np.log([self.bigrams_p[w] for w in self.bigrams]))
        t_ce = (-1 / len(self.trigrams)) * np.sum(np.log([self.trigrams_p[w] for w in self.trigrams]))
        print(u_ce, b_ce, t_ce)

    def classify(self, text):
    	text_unigrams = list(text)
    	text_bigrams = list(nltk.bigrams(text))
    	text_trigrams = list(nltk.trigrams(text))

    	text_unigram_p = reduce(lambda x,y: x*y, [self.smoothed_unigrams_p[w] for w in text_unigrams])
    	text_bigram_p = reduce(lambda x,y: x*y, [self.smoothed_bigrams_p[w] for w in text_bigrams])
    	text_trigram_p = reduce(lambda x,y: x*y, [self.smoothed_trigrams_p[w] for w in text_trigrams])
    	
    	print(text_unigram_p, text_bigram_p, text_trigram_p)

    def generate_text(self, word, gram='trigram'):
    	if(gram == 'unigram'):
	    	cfd = nltk.ConditionalFreqDist(self.smoothed_unigrams)
	    elif(gram == 'bigram'):
	    	cfd = nltk.ConditionalFreqDist(self.smoothed_bigrams)
	   	else
	   		cfd = nltk.ConditionalFreqDist(self.smoothed_trigrams)

	   	for i in range(100):
	   		print(word)
	   		word = cfd[word].max()
	   		     