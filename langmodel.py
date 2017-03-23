from collections import Counter
from functools import reduce
from copy import deepcopy

import numpy as np
import nltk

import random
import itertools
import bisect

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
		return (u_ce, b_ce, t_ce)

	def classify(self, text):
		text_unigrams = list(text)
		text_bigrams = list(nltk.bigrams(text))
		text_trigrams = list(nltk.trigrams(text))

		text_unigram_p = reduce(lambda x,y: x*y, [self.smoothed_unigrams_p[w] for w in text_unigrams])
		text_bigram_p = reduce(lambda x,y: x*y, [self.smoothed_bigrams_p[w] for w in text_bigrams])
		text_trigram_p = reduce(lambda x,y: x*y, [self.smoothed_trigrams_p[w] for w in text_trigrams])
		
		return (text_unigram_p, text_bigram_p, text_trigram_p)

	def unigram_gen(self):
		for i in range(100):
			choices, weights = zip(*self.unigrams_c.items())
			cumulate = list(itertools.accumulate(weights))
			x = random.random() * cumulate[-1]
			word = choices[bisect.bisect(cumulate, x)]
			print(word, end=" ")

	def bigram_gen(self):
		start = self.unigrams[random.randrange(0, len(self.unigrams) - 1)]
		word = start

		cfd = nltk.ConditionalFreqDist(self.bigrams)

		for i in range(100):
			print(word, end=" ")
			choices, weights = zip(*cfd[word].items())
			cumulate = list(itertools.accumulate(weights))
			x = random.random() * cumulate[-1]
			word = choices[bisect.bisect(cumulate, x)]

	def trigram_gen(self):
		index = random.randrange(0, len(self.unigrams) - 1)
		word1 = self.unigrams[index]
		word2 = self.unigrams[index+1]
		print("%s %s" % (word1, word2), end=" ")

		for i in range(100):
			# issue fixed. dont worry :D
			choice_list, counts = zip(*{gram[2]: self.trigrams_c[gram] for gram in self.trigrams_c if gram[0] == word1 and gram[1] == word2}.items())
			cumulate = list(itertools.accumulate(counts))
			x = random.random() * cumulate[-1]
			word = choice_list[bisect.bisect(cumulate, x)]
			print(word, end=" ")
			word1 = word2
			word2 = word

	def generate_text(self, gram=2):
		if(gram == 0):
			self.unigram_gen()
		elif(gram == 1):
			self.bigram_gen()
		else:
			self.trigram_gen()
		print()
