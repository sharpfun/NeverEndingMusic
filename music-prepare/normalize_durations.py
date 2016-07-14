import re
import json

file_path = "dataset/syllables_rhythm_notes.json"
normalized_outfile = 'dataset/normalized_syllables_rhythm_notes.json'

divisions_set = set()


def clean_word(source):
    return "".join(re.findall("[a-z]+", source.lower()))


def lcm(x, y):
    tmp = x
    while (tmp % y) != 0:
        tmp += x
    return tmp


def lcmm(arr):
    return reduce(lcm, arr)


with open(file_path) as f:
    sheets = json.loads(f.read())
    for sheet in sheets:
        divisions_set.add(sheet["duration_divisions"])

    lcm_division = lcmm(list(divisions_set))

    new_sheets = []
    for sheet in sheets:
        new_durations = []
        new_syllables = []
        new_pitches = []

        mult_durations_by = lcm_division / sheet["duration_divisions"]
        for i in range(len(sheet["durations"])):
            syllable = clean_word(sheet["syllables"][i])
            if syllable:
                new_durations.append(sheet["durations"][i]*mult_durations_by)
                new_syllables.append(syllable)
                new_pitches.append(sheet["pitches"][i])

        new_sheets.append({
            "syllables": new_syllables,
            "durations": new_durations,
            "pitches": new_pitches,
            "duration_divisions": lcm_division,
            "file_path": sheet["file_path"]
        })

    with open(normalized_outfile, 'w') as outfile:
        json.dump(new_sheets, outfile)


