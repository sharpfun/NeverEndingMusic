__author__ = 'Steffen'

from nltk.corpus import cmudict
import re
import string
from nltk.tokenize import word_tokenize

class CmuDict:
    def __init__(self):
        self.dict = cmudict.dict()
        self.unknown_dict = {}
        for key in self.dict.keys():
            if "'" in key:
                self.unknown_dict[key.replace("'", '')] = key
            if key.endswith('ing'):
                self.unknown_dict[key.replace('ing', 'in')] = key
            if 'every' in key:
                self.unknown_dict[key.replace('every', 'evry')] = key

    def stress(self, word, cut_by_syllables=False):
        lowercase = word.replace('-', '').lower().rstrip(string.punctuation)

        syllables_len = word.count('-') + 1

        if len(lowercase) <= 0:
            return '4'
        if lowercase in self.dict:
            out = ''.join(self.dict[lowercase][0])
            out = re.sub(r'[^0-9]', '', out)
        elif lowercase in self.unknown_dict:
            out = ''.join(self.dict[self.unknown_dict[lowercase]][0])
            out = re.sub(r'[^0-9]', '', out)
        else:
            print lowercase
            out = '3'*syllables_len

        if cut_by_syllables:
            if len(out) > syllables_len:
                out = out[:syllables_len]
            if len(out) < syllables_len:
                out += '0' * (syllables_len - len(out))

        return out

    def stress_syllable_list(self, syllables):
        result = ''
        word = ''
        has_to_parse = False
        for syllable in syllables:
            if syllable.endswith("-"):
                word += syllable.replace("-", "") + "-"
                has_to_parse = True
            else:
                word += syllable.replace("-", "")
                result += self.stress(word, cut_by_syllables=True)
                word = ''
                has_to_parse = False

        if has_to_parse:
            result += self.stress(word[:-1], cut_by_syllables=True)

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