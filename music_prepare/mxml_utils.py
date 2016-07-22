def create_mxml(syllables, syllables_decoded, durations, pitches, durations_division_size=26880):
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

        duration_type_list = {
            'long': 16,
            'breve': 8,
            'whole': 4,
            'half': 2,
            'quarter': 1,
            'eighth': 0.5,
            '16th': 0.25,
            '32nd': 0.125,
            '64th': 0.625,
            '128th': 0.03125,
            '256th': 0.015625
        }

        duration_type_xml = ''

        current_duration = float(durations[i])

        for key, value in duration_type_list.iteritems():
            if current_duration/durations_division_size == value:
                duration_type_xml = '<type>%s</type>' % key
            elif current_duration / durations_division_size == value * 1.5:
                duration_type_xml = '<type>%s</type><dot/>' % key
            elif current_duration / durations_division_size == value * 1.75:
                duration_type_xml = '<type>%s</type><dot/><dot/>' % key
            elif (current_duration*1.5) / durations_division_size == value:
                duration_type_xml = '<type>%s</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>' % key
            if duration_type_xml != '':
                break

        output += """<measure number="%s">
            <attributes>
                <divisions>%s</divisions>
            </attributes>
            <note default-y="-30.00" default-x="12.00">
                <pitch>%s</pitch>
                <duration>%s</duration>
                %s
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
        </measure>""" % (i+2, durations_division_size, pitch_xml, durations[i], duration_type_xml, syllables[i], syllables_decoded[i].replace("<", "&lt;").replace(">", "&gt;"))

    output += """</part>
    </score-partwise>"""

    return output