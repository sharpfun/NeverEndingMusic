if __name__ == "__main__":
    corpus_path = 'dataset/finalfile.txt'
    seqlength = 100
    prepared_corpus_out = 'dataset/finalfile-seqlen-'+str(seqlength)+'.txt'

    with open(corpus_path) as f:
        corpus = f.read()

    sheets = []

    for sheet in corpus.split("X:1"):
        only_notes = []
        for line in sheet.split("\n"):
            if len(line) > 2 and line[1] != ':' and not (line[0] > 'G' and line[0] < 'Z'):
                only_notes.append(line)

        rhythm = []

        notes = "".join(only_notes)
        prev_was_note = False
        for i in range(len(notes)):
            cur = notes[i]
            if (cur >= 'A' and cur <= 'G') or (cur >= 'a' and cur <= 'g'):
                rhythm.append('A')
                prev_was_note = True
            elif cur == 'z':
                rhythm.append('z')
                prev_was_note = True
            elif cur == "'" or cur == ",":
                prev_was_note = True
            elif cur == '>' or cur == '<':
                rhythm.append(cur)
            elif cur == '/':
                rhythm.append(cur)
                prev_was_note = True
            elif cur >= '2' and cur <= '9' and prev_was_note:
                rhythm.append(cur)
            else:
                prev_was_note = False

        if len(''.join(set(rhythm))) > 1:
            print "".join(rhythm)
            sheets.append(rhythm)

    sheets = "".join(["".join(x[:seqlength+1]) for x in sheets if len(x) > seqlength])

    f = open(prepared_corpus_out, 'w')
    f.write(sheets)  # python will convert \n to os.linesep
    f.close()
