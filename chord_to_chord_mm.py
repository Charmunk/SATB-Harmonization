'''
This is an implementation of a markov model that predicts the next chord given
the previous chord. We decided to normalize all songs to C Major because this
reduces the space we need to deal with. Chord transitions don't depend on the
key that the song is in.
'''

# for every song in dataset/1/[song_file]/mix.mid
#     midi_parse it to get chord array for song ('Dm', 'GM', 'CM')
#     save chord array to dataset/1/[song_file]/chords.json

import os
import json

from midi_parser import midi_parse 

dataset_dir = "dataset/1"

for song_folder in os.listdir(dataset_dir):
    song_path = os.path.join(dataset_dir, song_folder, "mix.mid")
    chords_json_path = os.path.join(dataset_dir, song_folder, "chords.json")
    if os.path.exists(song_path):
        try:
            chords = midi_parse(song_path)
            with open(chords_json_path, "w") as f:
                json.dump(chords, f)
        except Exception as e:
            print(f"Error processing {song_path}: {e}")

def chord_to_chord_mm(dataset_dir):
    '''
    Initialize chords (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)^2 transition matrix. 

    loop through every song in the dataset:
        for that song, midi_parse it to get chord array for song ('Dm', 'GM', 'CM')
            loop through every chord in the song from the second chord to the final
            chord. 
                Count transitions: Add a 1 to transition matrix for every chord
                transition.
    Normalize the matrix so every column adds up to 1.
    '''
    roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_to_idx = {r: i for i, r in enumerate(roots)}
    n = len(roots)

    transition_matrix = [[0] * n for _ in range(n)]

    for song_folder in os.listdir(dataset_dir):
        chords_json_path = os.path.join(dataset_dir, song_folder, "chords.json")
        if not os.path.exists(chords_json_path):
            continue
        with open(chords_json_path, "r") as f:
            chords = json.load(f)
        for i in range(1, len(chords)):
            prev_root = chords[i-1][:2] if len(chords[i-1]) > 1 and chords[i-1][1] == '#' else chords[i-1][0]
            curr_root = chords[i][:2]   if len(chords[i])   > 1 and chords[i][1]   == '#' else chords[i][0]
            if prev_root in root_to_idx and curr_root in root_to_idx:
                transition_matrix[root_to_idx[prev_root]][root_to_idx[curr_root]] += 1

    # Normalize each row to sum to 1
    for row in range(n):
        row_sum = sum(transition_matrix[row][col] for col in range(n))
        if row_sum > 0:
            for col in range(n):
                transition_matrix[row][col] /= row_sum

    return transition_matrix