__author__ = 'Steffen'

import lyrics

def get_songs():
    songlist = []
    with open('unique_tracks.txt') as f:
        contents = f.read()
        for line in contents.split('\n'):
            parts = line.split('<SEP>')
            try:
                artist = parts[2]
                title = parts[3]
                songlist.append((artist,title))
            except Exception:
                print line
    return songlist

start = 5840

songlist = get_songs()[5840:]
x = lyrics.LyricsPlugin()
written = 0
for idx, (artist,title) in enumerate(songlist):
    lyrics = x.get_lyrics(artist, title)
    if lyrics is not None:
        try:
            lyrics = lyrics.decode('ascii')
            with open('lyrics_out.txt', 'a') as f:
                f.write(lyrics + '\n')
                f.close()
                written += 1
                print artist + ' - ' + title
                print 'Written: ' + str(written)
                print 'Index: ' + str(start + idx)

        except Exception as ex:
            print ex.message