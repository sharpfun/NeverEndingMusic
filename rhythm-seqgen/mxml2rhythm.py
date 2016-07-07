import os
from bs4 import BeautifulSoup as bs

folder = "/home/kroman/Downloads/nn/Wikifonia/"

all_notes = []
seq_len = 100
prepared_corpus_out = 'wikifonia-seqlen-100.txt'

for root, dirs, files in os.walk(folder):
    for xml_file in files:
        if xml_file.endswith(".xml"):
            file_notes = []
            file_path = os.path.join(root, xml_file)
            with open(file_path) as f:
                xml = bs(f.read(), "xml")

                print file_path

                for note in xml.find_all("note"):
                    bit = None
                    if note.pitch and note.duration:
                        bit = 'N'+note.duration.text
                    if note.rest and note.duration:
                        bit = 'R'+note.duration.text
                    if bit:
                        file_notes.append(bit)

                if len(file_notes) > seq_len:
                    all_notes = all_notes+file_notes[:seq_len+1]

print len(all_notes)

f = open(prepared_corpus_out, 'w')
f.write(",".join(all_notes))
f.close()

