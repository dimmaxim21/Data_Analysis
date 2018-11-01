import pandas as pd
from my_library.PLSA import PLSA
from texts.StopWords import StopWords
from texts.Corpus import Corpus
from os import getcwd

sw = StopWords("E:/GitProjects/lab3/texts/stopwords.dic")
sw.loadStopWordsFromFile()

data = pd.read_csv("E:/GitProjects/lab3/data/lenta_ru.csv")
documents = data["text"].tolist()
tags = data["tags"].tolist()

corpus = Corpus()
corpus.loadCorpusFromList(documents, tags)

K = 10    # number of topic
maxIteration = 30
threshold = 10.0
topicWordsNum = 10
docTopicDist = 'docTopicDistribution.txt'
docTopicDist = getcwd() + "/results/" + docTopicDist
topicWordDist = 'topicWordDistribution.txt'
topicWordDist = getcwd() + "/results/" + topicWordDist
dictionary = 'dictionary.dic'
dictionary = getcwd() + "/results/" + dictionary
topicWords = 'topics.txt'
topicWords = getcwd() + "/results/" + topicWords

plsa = PLSA(corpus, sw, K, maxIteration, threshold, topicWordsNum, docTopicDist, topicWordDist, dictionary, topicWords)
plsa.EM_Algo()
