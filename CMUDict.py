__author__ = 'Steffen'

from nltk.corpus import cmudict
import re
import string
from nltk.tokenize import word_tokenize

class CmuDict:
    def __init__(self):
        self.dict = cmudict.dict()

    def stress(self, word):
        lowercase = word.replace('-', '').lower().rstrip(string.punctuation)
        if len(lowercase) <= 0:
            return '4'
        if lowercase in self.dict:
            out = ''.join(self.dict[lowercase][0])
            out = re.sub(r'[^0-9]', '', out)
        else:
            print lowercase
            out = '3'*(word.count('-')+1)

        return out

    def stress_syllable_list(self, syllables):
        result = ''
        word = ''
        for syllable in syllables:
            if syllable.endswith("-"):
                word += syllable
            else:
                word += syllable
                result += self.stress(word)
                word = ''
        return result

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