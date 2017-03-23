from langmodel import LangaugeModel
import nltk

C1 = LangaugeModel('austen-emma.txt')
C2 = LangaugeModel('edgeworth-parents.txt')
C3 = LangaugeModel('whitman-leaves.txt')

ug_set = set(C1.unigrams) | set(C2.unigrams) | set(C3.unigrams)
bg_set = set(C1.bigrams) | set(C2.bigrams) | set(C3.bigrams)
tg_set = set(C1.trigrams) | set(C2.trigrams) | set(C3.trigrams)

C1.smooth(ug_set, bg_set, tg_set)
C2.smooth(ug_set, bg_set, tg_set)
C3.smooth(ug_set, bg_set, tg_set)

C1.calculate_cross_entropy_values()
C2.calculate_cross_entropy_values()
C3.calculate_cross_entropy_values()

# C2.classify(nltk.corpus.gutenberg.sents('austen-emma.txt')[100])

C1.generate_text('The')
