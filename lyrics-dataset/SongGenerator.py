__author__ = 'Steffen'

from CMUDict import CmuDict
from gen import gen

text = gen(corpus='lyrics_out.txt')
print text
stressDict = CmuDict()
print stressDict.text(text)