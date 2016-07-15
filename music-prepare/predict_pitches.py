import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from blocks.extensions.saveload import load
from theano import function
import numpy
from dataset import T_H5PYDataset
from datetime import datetime


def gen():
    source_path = 'dataset/normalized_syllables_rhythm_notes.json-seqlen-100.hdf5'

    train_dataset = T_H5PYDataset(source_path, which_sets=('train',))

    main_loop = load('./checkpoint-pitches.zip')

    model = main_loop.model

    print [x.name for x in model.shared_variables]

    print [x.name for x in model.variables]

    tensor_initial = [x for x in model.shared_variables if x.name == "initial_state"][0]
    tensor_hidden_states = [x for x in model.intermediary_variables if x.name == "hidden_apply_states"][0]
    tensor_x = [x for x in model.variables if x.name == "durations"][0]
    tensor_y = [x for x in model.variables if x.name == "ndim_softmax_apply_output"][0]

    predict_fun = function([tensor_x], tensor_y, updates=[
        (tensor_initial, tensor_hidden_states[0][0]),
    ])

    input_str = "love-looks-not-with-the-eyes-but-with-the-mind-and-there-fore-is-winged-cupid-pain-ted-blind"

    input_str_split = input_str.split("-")

    input_str_arr = train_dataset.syllables_encode(input_str_split)

    input_str_arr_decoded = train_dataset.syllables_decode(input_str_arr)

    input_durations = [13440, 26880, 17920, 13440, 26880, 26880, 26880, 13440, 13440, 80640, 13440, 13440, 26880, 53760,
                       26880, 26880, 13440, 13440, 13440]
    #[80640, 26880, 40320, 13440, 26880, 26880, 26880, 13440, 26880, 80640, 13440, 13440, 40320, 26880, 26880, 26880, 13440, 13440, 13440]
    input_durations_arr = train_dataset.durations_encode(input_durations)

    print input_str_arr
    print input_durations_arr

    predictions = []
    import time
    numpy.random.seed(int(time.time()))
    for i in range(len(input_durations_arr)):
        input_char = numpy.zeros((1, 1), dtype=numpy.int32)
        input_char[0][0] = input_durations_arr[i]
        predictions.append(numpy.random.choice(train_dataset.pitches_vocab_size(), 1, p=predict_fun(input_char)[0])[0])

    output = """<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 2.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
    <score-partwise version="2.0">
        <work>
            <work-number></work-number>
            <work-title></work-title>
        </work>
        <movement-number></movement-number>
        <movement-title>Rural Life</movement-title>
        <identification>
            <creator type="composer">A. MacKenzie Davidson</creator>
            <creator type="poet"></creator>
            <rights>All Rights Reserved</rights>
            <encoding>
                <software>MuseScore 1.2</software>
                <encoding-date>2013-05-29</encoding-date>
                <software>ProxyMusic 2.0 c</software>
            </encoding>
            <source>http://wikifonia.org/node/22433/revisions/29715/view</source>
        </identification>
        <defaults>
            <scaling>
                <millimeters>7.2319</millimeters>
                <tenths>40</tenths>
            </scaling>
            <page-layout>
                <page-height>1642.72</page-height>
                <page-width>1161.52</page-width>
                <page-margins type="both">
                    <left-margin>105</left-margin>
                    <right-margin>70</right-margin>
                    <top-margin>42</top-margin>
                    <bottom-margin>48</bottom-margin>
                </page-margins>
            </page-layout>
        </defaults>
        <credit page="1">
            <credit-words font-size="24" default-y="1600.72" default-x="580.76" justify="center" valign="top">Rural Life</credit-words>
        </credit>
        <credit page="1">
            <credit-words font-size="14" default-y="1545.41" default-x="580.76" justify="center" valign="top">Intro</credit-words>
        </credit>
        <credit page="1">
            <credit-words font-size="12" default-y="1532.92" default-x="1091.52" justify="right" valign="top">A. MacKenzie Davidson</credit-words>
        </credit>
        <part-list>
            <score-part id="P1">
                <part-name></part-name>
                <score-instrument id="P1-I1">
                    <instrument-name></instrument-name>
                </score-instrument>
                <midi-instrument id="P1-I3">
                    <midi-channel>1</midi-channel>
                    <midi-program>49</midi-program>
                    <volume>79.5276</volume>
                    <pan>0</pan>
                </midi-instrument>
            </score-part>
        </part-list>
        <part id="P1">
            <measure width="180.23" number="1">
                <print>
                    <system-layout>
                        <system-margins>
                            <left-margin>-0.00</left-margin>
                            <right-margin>-0.00</right-margin>
                        </system-margins>
                        <top-system-distance>229.34</top-system-distance>
                    </system-layout>
                </print>
                <attributes>
                    <divisions>0</divisions>
                    <key>
                        <fifths>0</fifths>
                        <mode>major</mode>
                    </key>
                    <time>
                        <beats>3</beats>
                        <beat-type>4</beat-type>
                    </time>
                    <clef>
                        <sign>G</sign>
                        <line>2</line>
                    </clef>
                </attributes>
            </measure>"""

    pitches = train_dataset.pitches_decode(predictions)
    print "Predict:"
    print pitches

    for i in range(len(pitches)):
        pitch = pitches[i]
        if len(pitch) == 2:
            pitch_xml = """<step>%s</step>
                    <octave>%s</octave>""" % (pitch[0], pitch[1])
        elif len(pitch) == 3:
            pitch_xml = """<step>%s</step>
                    <octave>%s</octave>
                    <alter>%s</alter>""" % (pitch[0], pitch[2], "-1" if pitch[1] == "b" else "1")
        else:
            raise NotImplementedError

        output += """<measure number="%s">
            <attributes>
                <divisions>26880</divisions>
            </attributes>
            <note default-y="-30.00" default-x="12.00">
                <pitch>%s</pitch>
                <duration>%s</duration>
                <voice>1</voice>
                <stem>up</stem>
                <lyric number="1">
                    <text>%s</text>
                    <extend/>
                </lyric>
                <lyric number="2">
                    <text>%s</text>
                    <extend/>
                </lyric>
            </note>
        </measure>""" % (i+2, pitch_xml, input_durations[i], input_str_split[i], input_str_arr_decoded[i].replace("<", "&lt;").replace(">", "&gt;"))

    output += """</part>
    </score-partwise>"""

    return output


if __name__ == "__main__":
    output = gen()
    f = open('generated/rhythm/sample ' + str(datetime.utcnow()) + ".xml", 'w')
    f.write(output)
    f.close()