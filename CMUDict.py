__author__ = 'Steffen'

from nltk.corpus import cmudict
import re
import string
from nltk.tokenize import word_tokenize

class CmuDict:
    def __init__(self):
        self.dict = cmudict.dict()

    def stress(self, word):
        lowercase = word.lower().rstrip(string.punctuation)
        if len(lowercase) <= 0:
            return ''
        if lowercase in self.dict:
            out = ''.join(self.dict[lowercase][0])
            out = re.sub(r'[^0-9]', '', out)
        else:
            print lowercase
            out = '3'

        return out

    def sentence(self, sent):
        out = ''
        for word in word_tokenize(sent):
            out += self.stress(word)
        return out

    def text(self, text):
        out = ''
        for sent in text.splitlines():
            out += self.sentence(sent) + '\n'
        return out