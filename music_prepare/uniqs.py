import json

normalized_outfile = 'dataset/normalized_syllables_rhythm_notes.json'


with open(normalized_outfile) as f:
    durations_set = set()
    syllables_dict = dict()
    pitches_set = set()

    sheets = json.loads(f.read())
    for sheet in sheets:
        for i in range(len(sheet["durations"])):
            durations_set.add(sheet["durations"][i])

            syllable = sheet["syllables"][i]
            if syllable not in syllables_dict:
                syllables_dict[syllable] = set()
            syllables_dict[syllable].add(sheet["file_path"])
            pitches_set.add(sheet["pitches"][i])

    syllables_set = set()

    for key, value in syllables_dict.iteritems():
        if len(value) > 10:
            syllables_set.add(key)

    for sheet in sheets:
        for i in range(len(sheet["durations"])):
            if sheet["syllables"][i] not in syllables_set:
                sheet["syllables"][i] = "<unk>"

    all_durations = []
    all_syllables = []
    all_pitches = []

    for sheet in sheets:
        all_durations += sheet["durations"] + [0]
        all_syllables += sheet["syllables"] + ["<end_file>"]
        all_pitches += sheet["pitches"] + ["R"]

    print durations_set
    print len(durations_set)

    print syllables_set
    print len(syllables_set)

    print pitches_set
    print len(pitches_set)