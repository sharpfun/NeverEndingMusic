import re
import json
from CMUDict import CmuDict


file_path = "dataset/syllables_rhythm_notes.json"
normalized_outfile = 'dataset/normalized_syllables_rhythm_notes.json'

divisions_set = set()


def clean_word(source):
    return "".join(re.findall("[a-z\-]+", source.lower()))


def lcm(x, y):
    tmp = x
    while (tmp % y) != 0:
        tmp += x
    return tmp


def lcmm(arr):
    return reduce(lcm, arr)

note_to_val = {'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'C': 3, 'B#': 3, 'C#': 4, 'Db': 4, 'D': 5, 'D#': 6,
               'Eb': 6, 'E': 7, 'Fb': 7, 'E#': 8, 'F': 8, 'F#': 9, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 11}
val_to_note = {0: 'A', 1: 'Bb', 2: 'B', 3: 'C', 4: 'C#', 5: 'D', 6: 'Eb', 7: 'E', 8: 'F', 9: 'F#', 10: 'G', 11: 'G#'}

def transpose(fifths, note):
    note_name = note[:-1]
    octave = int(note[-1])
    note_val = note_to_val[note_name]

    shift_by = - ((int(fifths) * 7) % 12)
    new_note_val = (note_val + shift_by) % 12

    new_octave = int(octave)
    if note_val + shift_by - 3 < 0 and note_val >= 3:
        new_octave -= 1

    return val_to_note[new_note_val] + str(new_octave)

with open(file_path) as f:
    sheets = json.loads(f.read())
    for sheet in sheets:
        divisions_set.add(sheet["duration_divisions"])

    lcm_division = lcmm(list(divisions_set))

    new_sheets = []

    cmudict = CmuDict()

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
                new_pitches.append(transpose(sheet["fifths"], sheet["pitches"][i]))

        new_sheets.append({
            "syllables": new_syllables,
            "durations": new_durations,
            "pitches": new_pitches,
            "stress": list(cmudict.stress_syllable_list(new_syllables)),
            "fifths": sheet["fifths"],
            "duration_divisions": lcm_division,
            "file_path": sheet["file_path"]
        })

    with open(normalized_outfile, 'w') as outfile:
        json.dump(new_sheets, outfile)


