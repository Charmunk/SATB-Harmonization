'''
From 4-note format

note0,note1,note2,note3    --> maps to S, A, T, B so if you need just A, select the 2nd column.
72,64,55,48
72,64,55,48
72,64,55,48
72,64,55,48

creates

chorale_#_chords.csv.

TODO: chords need to be ordered so they are in root voicing. Otherwise,
pychord will shit the bed.

'''

from pychord import find_chords_from_notes
import csv
import os


def midi_to_note(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[midi_num % 12]


def note_names_for_chord(midis):
    """Pitch classes in S,A,T,B order, first occurrence only (pychord rejects duplicate names)."""
    seen = set()
    out = []
    for n in midis:
        pc = n % 12
        if pc in seen:
            continue
        seen.add(pc)
        out.append(midi_to_note(n))
    return out


# Walk chorale CSVs under this directory (train / valid / test).
dataset = os.path.dirname(os.path.abspath(__file__))

for root, dirs, files in os.walk(dataset):
    for file in files:
        if not file.endswith('.csv') or file.endswith('_chords.csv'):
            continue
        src_path = os.path.join(root, file)
        base, _ = os.path.splitext(file)
        out_path = os.path.join(root, f'{base}_chords.csv')
        print('making', out_path, 'from', src_path)
        with open(src_path, newline='') as f_in, open(out_path, 'w', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            next(reader, None)  # skip note0,note1,note2,note3 header
            for row in reader:
                if len(row) < 4:
                    continue
                midis = [int(row[i]) for i in range(4)]
                note_names = note_names_for_chord(midis)
                found = find_chords_from_notes(note_names) if note_names else []
                chord_str = str(found[0]) if found else ''
                writer.writerow([chord_str])
