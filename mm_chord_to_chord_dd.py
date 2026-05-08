import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
# from mm_part_to_part import *
import importlib
from lookup_delta_3d import *

def csv_to_tracks(file_path):
    """
    Reads a 4-column CSV and returns four 1D numpy arrays.
    
    Args:
        file_path (str): Path to the .csv file.
        
    Returns:
        tuple: (arr1, arr2, arr3, arr4) as 1D numpy arrays.
    """
    # unpack=True turns columns into individual arrays
    # skip_header=1 ignores the first row of text
    # delimiter=',' ensures we parse CSV format correctly
    S, A, T, B = np.genfromtxt(
        file_path, 
        delimiter=',', 
        unpack=True, 
        skip_header=1,
        dtype=int
    )
    return S, A, T, B

def get_chord_dict(dataset_dir):
    """
    Data-driven approach to enumerate all observed chords

    Args:
        dataset_dir (str): The directory containing the training data.

    Returns:
        chord_to_idx: dictionary encoding (A, T, B) --> idx 
        idx_to_chord: dictionary encoding idx --> (A, T, B) 
        chord_to_chord: max(idx)xmax(idx) state transition matrix for chords
    """
    songs = glob.glob('*.csv', root_dir=dataset_dir)
    songs = [s for s in songs if not 'd' in s] #remove all strings with 'd' in them (filters out chord csv)
    chord_to_idx = {}
    idx_to_chord = {}
    chord_idx_list = []
    idx = 0
    total_data = 0

    #can can shrink because they don't have access to all notes, just going to make a 128x128 mat
    counts_A = np.ones((128, 128), dtype='int64') *1/1000 #add an epsilon so no divide by zero
    counts_T = np.ones((128, 128), dtype='int64') *1/1000 #add an epsilon so no divide by zero
    counts_B = np.ones((128, 128), dtype='int64') *1/1000 #add an epsilon so no divide by zero

    for song in tqdm(songs, desc="Parsing CSVs", unit="song"):
        song_path = os.path.join(dataset_dir, song)

        if os.path.exists(song_path):
            try:
                _, A, T, B = csv_to_tracks(song_path)

            except Exception as e:
                tqdm.write(f"Error processing {song_path}: {e}")

            N = len(B) #how many datapoints we have
            total_data += N
            ATB = []
            for i in range(1, N):
                chord = (A[i], T[i], B[i])
                # make array of chords in order,  then make dict and use for transition model
                # question.. big array okay? like 100x100
                if chord not in chord_to_idx:
                    chord_to_idx[chord] = idx
                    idx_to_chord[idx] = chord
                    idx += 1
                chord_idx_list.append(chord_to_idx[chord]) # add chord idx to list of seen chords   
    chord_trans_counts = np.ones((idx, idx), dtype='int64')*1/1000 # add an epsilon so no divide by zero
    # for i in range(idx):
    #      chord_to_chord[]
    for i in range(1,len(chord_idx_list)):  #index thru all chords seen
        prev_chord_idx = chord_idx_list[i-1]
        chord_idx = chord_idx_list[i]
        chord_trans_counts[prev_chord_idx, chord_idx] += 1

        # double count chord transitions
        if prev_chord_idx != chord_idx:
            chord_trans_counts[prev_chord_idx, chord_idx] += 1000
            # print("chord mats")
            
        #     print(f"chord transition occured! from {prev_chord_idx} to {chord_idx}")
    chord_to_chord = np.zeros_like(chord_trans_counts)
    # just in case we decide to change their shapes, normalize individuallly
    for i in range(chord_to_chord.shape[0]):
        chord_to_chord[i, :] = chord_trans_counts[i, :] / np.sum(chord_trans_counts[i, :]) #normalize
    print(f" total chords seen: {total_data}")
    return chord_to_idx, idx_to_chord, chord_to_chord

def get_delta_3d_lookup_chord(train_dir, chord_to_idx, num_semitones=4):
    #returns the log probability matrix of P(delta | soprano_{n-1}, chord_n)
    #index by (chord, prev_sporano, delta)
    num_chords = len(chord_to_idx)
    lookup_table = np.ones((num_chords, 128, 2*num_semitones+2)) * 1/1000

    songs = glob.glob('*.csv', root_dir=train_dir)
    songs = [s for s in songs if not 'd' in s] #remove all strings with 'd' in them (filters out chord csv)

    # Iterate through every .csv in the train directory
    for songname in songs:
        # You can now use `csv_path` for each CSV file
        csv_path = os.path.join(train_dir, songname)
        
        S, A, T, B = csv_to_tracks(csv_path)
        
        # Combine into array of tuples [(S, A, T, B)]
        SATB_tuples = list(zip(S, A, T, B))
        for i in range(1, len(SATB_tuples)):
            #get current tuple
            s, a, t, b = SATB_tuples[i]
            # get previous soprano note
            prev_s = SATB_tuples[i-1][0]
            # get delta
            delta = s - prev_s
            # get chord
            chord_idx = chord_to_idx[(a,t,b)]
            # update lookup table
            lookup_table[int(chord_idx), int(prev_s), delta_2_index(delta)] += 1

        # normalize the lookup table (for each delta)
        for i in range(lookup_table.shape[2]):
            lookup_table[:, :, i] = lookup_table[:, :, i] / np.sum(lookup_table[:, :, i])
            
        return np.log(lookup_table) #returns the log probability
