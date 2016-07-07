__author__ = 'Steffen'

from gen import gen
from blocks.extensions import SimpleExtension
from datetime import datetime

class ShowOutput(SimpleExtension):

    def do(self, which_callback, *args):
        output = gen()
        f = open('generated/sample '+str(datetime.utcnow())+".xml", 'w')
        f.write(output)
        f.close()
