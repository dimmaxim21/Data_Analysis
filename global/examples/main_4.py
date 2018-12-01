import pandas as pd
from my_library.LDA import LDA
from texts.stop_words import StopWords
from texts.corpus import Corpus

sw = StopWords("E:/GitProjects/lab4/texts/stopwords.dic")
sw.load_stop_words_from_file()
sw.load_stop_word_from_nltk_lib()

data = pd.read_csv("E:/GitProjects/lab4/data/lenta_ru.csv")
documents = data["text"].tolist()

corpus = Corpus()
corpus.load_corpus_from_list(documents)

lda = LDA(corpus=corpus, stop_words=sw, K=20, alpha=0.5, beta=0.5, iterations=50)
lda.run()
print("\n", lda.worddist())
