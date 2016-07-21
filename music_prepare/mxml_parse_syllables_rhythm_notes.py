import os
from bs4 import BeautifulSoup as bs
import enchant
import re
import shutil

d = enchant.Dict("en_US")

#folder = "/home/kroman/Downloads/nn/Wikifonia/Arrangement by John & Annie - Advent medley 2.mxl_FILES/"
#folder = "/home/kroman/Downloads/nn/Wikifonia/"
folder = 'C:\Users\Steffen\NeverEndingMusic\NeverEndingMusic\music_prepare\Wikifonia\'

all_notes = []
seq_len = 100
prepared_corpus_out = 'wikifonia-seqlen-100.txt'


def clean_word(source, extra_clean=False):
    if extra_clean:
        return ''.join([x for x in source if x not in ",!?.;'"]).lower()
    else:
        return ''.join([x for x in source if x not in ',!?.;']).lower()


def iterate_over_all_music_files(xml_parser):
    for root, dirs, files in os.walk(folder):
        for xml_file in files:
            if xml_file.endswith(".xml"):
                file_path = os.path.join(root, xml_file)
                with open(file_path) as f:
                    xml = bs(f.read(), "xml")

                    xml_parser(xml, file_path)


def mxml_print_syllables(xml, file_path):
    res = ""
    song_has_wrong_align = False

    for lyric in xml.find_all(name="lyric", attrs={"number": "1"}):
        if not lyric.find("text"):
            res += " _ "
            song_has_wrong_align = True
            continue

        syllable = clean_word(lyric.find("text").text)

        res += syllable

        if lyric.syllabic and lyric.syllabic.text == 'begin':
            res += ' - '
        else:
            res += ' '

    if song_has_wrong_align:
        res = "####### " + res
    print res


def mxml_is_english_song(xml, file_path, print_non_english_word=True):
    all_words = 0
    english_words = 0

    prev_part = ''

    for lyric in xml.find_all(name="lyric", attrs={"number": "1"}):
        if not lyric.find("text"):
            continue
        if not lyric.syllabic:
            all_words = 0
            english_words = 0
            break
        prev_part += lyric.find("text").text
        all_words += 1
        if lyric.syllabic.text in ['single', 'end']:
            word = ''.join([x for x in prev_part if x not in ',!?.;'])
            if word and d.check(word): # or len(d.suggest(word))
                english_words += 1
            prev_part = ''

    if english_words > 10:
        print "PERCENTAGE %s" % ((1.0*english_words)/all_words)

    return english_words > 10 and english_words/0.42 > all_words


class SongsCounter:
    music_all = 0
    songs_all = 0
    songs_english = 0


def mxml_print_english_songs(xml, file_path):
    print file_path
    SongsCounter.music_all += 1
    if xml.find(name="lyric", attrs={"number": "1"}) is not None:
        SongsCounter.songs_all += 1
    mxml_print_syllables(xml, file_path)
    if mxml_is_english_song(xml, file_path, False):
        SongsCounter.songs_english += 1
        print "YYYYYYYYYYYYYYYYYYYYYYYYYYY"
    else:
        print "NNNNNNNNNNNNNNNNNNNNNNNNNNN"


def mxml_delete_non_english_songs(xml, file_path):
    print file_path
    if not mxml_is_english_song(xml, file_path, False):
        print "YYYYYYYYYYYYYYYYYYYYYYYYYYY"
    else:
        print "NNNNNNNNNNNNNNNNNNNNNNNNNNN"
        shutil.rmtree(os.path.dirname(os.path.realpath(file_path)))


json_arr = []

def mxml_parse_syllables_rhythm_pitch(xml, file_path):
    print file_path

    syllables_arr = []
    durations_arr = []
    pitches_arr = []

    fifths = xml.find(name="key").fifths.text

    for note in xml.find_all(name="note"):
        lyric = note.find(name="lyric", attrs={"number": "1"})
        if not lyric or not lyric.find("text") or not note.pitch or not note.duration:
            continue
        syllable = clean_word(lyric.find("text").text, True)
        if lyric.syllabic and lyric.syllabic.text in ['begin', 'middle']:
            if not syllable.strip().endswith('-'):
                syllable += '-'

        duration = note.duration.text
        alter = ''
        if note.pitch.alter:
            if note.pitch.alter.text == '-1':
                alter = 'b'
            if note.pitch.alter.text == '1':
                alter = '#'
        pitch = note.pitch.step.text + alter + note.pitch.octave.text

        syllables_arr.append(syllable)
        durations_arr.append(int(duration))
        pitches_arr.append(pitch)

    all_divisions = xml.find_all(name="divisions")
    uniq_divisions = set()
    for division in all_divisions:
        uniq_divisions.add(division.text)
    if len(uniq_divisions) != 1:
        0/0

    json_arr.append({
        "file_path": file_path,
        "syllables": syllables_arr,
        "durations": durations_arr,
        "pitches": pitches_arr,
        "duration_divisions": int(xml.find(name="divisions").text),
        "fifths": fifths
    })


import json
if __name__ == "__main__":
    iterate_over_all_music_files(mxml_parse_syllables_rhythm_pitch)
    with open('dataset/syllables_rhythm_notes.json', 'w') as outfile:
        json.dump(json_arr, outfile)
