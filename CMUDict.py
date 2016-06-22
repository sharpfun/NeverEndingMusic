__author__ = 'Steffen'

from nltk.corpus import cmudict
import re
import string

class CmuDict:
    def __init__(self):
        self.dict = cmudict.dict()

    def stress(self, word):
        lowercase = word.lower().rstrip(string.punctuation)

        if lowercase in self.dict:
            out = ''.join(self.dict[lowercase][0])
            out = re.sub(r'[^0-9]', '', out)
        else:
            print lowercase
            out = '3'

        return out
