from langmodel import LangaugeModel
import nltk
import random

C1 = None
C2 = None
C3 = None

while True:
	print("1. Build model")
	print("2. Cross entropy")
	print("3. Classify Author")
	print("4. Get Probability")
	print("5. Generate Text")
	print("6. Exit")
	ch = int(input())

	if(ch == 6):
		exit(0)

	elif(ch == 1):
		print("Building austen emmma")
		C1 = LangaugeModel('austen-emma.txt')
		print("Building edgeworth parents")
		C2 = LangaugeModel('edgeworth-parents.txt')
		print("Building whitman leaves")
		C3 = LangaugeModel('whitman-leaves.txt')

		ug_set = set(C1.unigrams) | set(C2.unigrams) | set(C3.unigrams)
		bg_set = set(C1.bigrams) | set(C2.bigrams) | set(C3.bigrams)
		tg_set = set(C1.trigrams) | set(C2.trigrams) | set(C3.trigrams)

		print("Smoothing...")
		C1.smooth(ug_set, bg_set, tg_set)
		C2.smooth(ug_set, bg_set, tg_set)
		C3.smooth(ug_set, bg_set, tg_set)
		print("Done")

	elif(ch == 2):
		print("Calculating cross entropy")
		print("%s\t%s\t%s" % ("unigram", "bigram", "trigram"))
		print("C1")
		print(C1.calculate_cross_entropy_values())
		print("C2")
		print(C2.calculate_cross_entropy_values())
		print("C3")
		print(C3.calculate_cross_entropy_values())

	elif(ch == 3):
		nsent = 1
		book = "austen-emma.txt"
		sents = list(nltk.corpus.gutenberg.sents(book))
		x = random.randint(0, len(sents) - nsent)
		print("random number: ", x)
		text = []
		for i in range(nsent):
			text = text + sents[x+i]
		print("Text chosen from ", book)
		a1 = C1.classify(text)
		a2 = C2.classify(text)
		a3 = C3.classify(text)

		print("Austen\tEdgeworth\tWhitman")
		print("Unigrams")
		print(a1[0], a2[0], a3[0])
		print("Bigrams")
		print(a1[1], a2[1], a3[1])
		print("Trigrams")
		print(a1[2], a2[2], a3[2])

	elif(ch == 4):
		print("0. Austen 1. Edgeworth 2. Whitman")
		a = int(input())
		book = [C1, C2, C3][a]
		print("0. Unigram 1. Bigram 2. Trigram")
		b = int(input())
		if(b == 0):
			print("enter string: ")
			s = input().strip()
			print(book.smoothed_unigrams_p[s])
		elif(b==1):
			print("enter string: ")
			s = input().strip().split()
			print(book.smoothed_bigrams_p[(s[0], s[1])])
		elif(b==2):
			print("enter string: ")
			s = input().strip().split()
			print(book.smoothed_trigrams_p[(s[0], s[1], s[2])])

	elif(ch == 5):
		print("0. Austen 1. Edgeworth 2. Whitman")
		a = int(input())
		book = [C1, C2, C3][a]
		print("0. Unigram 1. Bigram 2. Trigram")
		b = int(input())
		book.generate_text(b)
