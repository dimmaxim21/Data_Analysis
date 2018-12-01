import codecs
import random
from unittest import TestCase
from texts.stop_words import StopWords


class TestStopWords(TestCase):

    def setUp(self):
        self.sw = StopWords("E:/GitProjects/lab4/texts/stopwords.dic")
        self.sw.load_stop_words_from_file()
        file = codecs.open("E:/GitProjects/lab4/texts/stopwords.dic", 'r', 'utf-8')
        self.word_list = [line.strip() for line in file]
        file.close()

    def test_get_stop_words(self):
        self.assertEqual(self.sw.get_stop_words(), self.word_list)

    def test_is_top_word(self):
        self.assertEqual(self.sw.is_stop_word(
            self.sw.get_stop_words()[random.randint(0, len(self.sw.get_stop_words()))]), True)
