import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def csv_to_tracks(file_path):
    """
    Reads a 4-column CSV and returns four 1D numpy arrays.
    
    Args:
        file_path (str): Path to the .csv file.
        
    Returns:
        tuple: (S, A, T, B) as 1D numpy arrays.
    """
    # unpack=True turns columns into individual arrays
    # skip_header=1 ignores the first row of text
    # delimiter=',' ensures we parse CSV format correctly
    S, A, T, B = np.genfromtxt(
        file_path, 
        delimiter=',', 
        unpack=True, 
        skip_header=1
    )
    
    return S, A, T, B


def train_bass_to_bass_markov(dataset_dir):
    """
    arguments: a string with containing the training directory
    returns: a normalized B -> B markov transition matrix
    """
    songs = glob.glob('*.csv', root_dir=dataset_dir)
    songs = [s for s in songs if not 'd' in s] #remove all strings with 'd' in them (filters out chord csv)

    #can can shrink because they don't have access to all notes, just going to make a 128x128 mat
    counts = np.ones((128, 128), dtype='int64') *1/1000 #add an epsilon so no divide by zero

    for song in tqdm(songs, desc="Parsing CSVs", unit="song"):
        song_path = os.path.join(dataset_dir, song)

        if os.path.exists(song_path):
            try:
                _, _, _, B = csv_to_tracks(song_path)

            except Exception as e:
                tqdm.write(f"Error processing {song_path}: {e}")

            N = len(B) #how many datapoints we have

            for i in range(1, N):
                #if B[i-1] != B[i]:
                    counts[int(B[i-1]), int(B[i])] += 1
    
    transition_matrix = np.zeros_like(counts)
    for i in range(transition_matrix.shape[0]):
        transition_matrix[i, :] = counts[i, :] / np.sum(counts[i, :]) #normalize

    return transition_matrix