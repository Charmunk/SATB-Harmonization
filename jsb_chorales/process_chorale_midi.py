'''
From 4-note format

note0,note1,note2,note3    --> maps to S, A, T, B so if you need just A, select the 2nd column.
72,64,55,48
72,64,55,48
72,64,55,48
72,64,55,48

creates 

chorale_#_chords.csv.

'''

from pychord import find_chords_from_notes
import os

# Example MIDI note numbers for a C major chord (C4, E4, G4)
# 60 = C4, 64 = E4, 67 = G4
midi_notes = [60, 64, 67]

# 1. Convert MIDI note numbers to note names (simplest way)
def midi_to_note(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[midi_num % 12]

# loop through every file in dataset.
dataset = 'dataset/jsb_chorales'

# For every file in dataset (this means in the folders 'test', 'train', 'valid'),
# create a new csv for that file, appending [original_file]_chords.csv.

for root, dirs, files in os.walk(dataset):
    for file in files:
        # Join root and file to get the full path
        print("making chords.csv for", os.path.join(root, file))
        note_names = [midi_to_note(n) for n in midi_notes]
        chord = find_chords_from_notes(note_names)[0]
        
