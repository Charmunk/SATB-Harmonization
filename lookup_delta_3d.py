import numpy as np
import glob
import os

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
        skip_header=1
    )
    
    return S, A, T, B

def delta_2_index(delta, num_semitones=4):
    """
    turns sometimes negative valued deltas into a consistent index, including processing rests
    [ 0.  1.  2.  3.  4. -4. -3. -2. -1.]
    if non-negative, index is delta
    if negative, index is  2 x (number of semitones) + 1 + delta (e.g. -1 -> 8, -4 -> 5)
    if rest, maps to 2 x (number of semitones) + 1
    arguments: takes in an integer delta (-num_semintones <= delta <= num_semitones) unless it's a rest
    returns: the index
    """

    if num_semitones >= delta >= 0:
        return int(delta)
    elif -1*num_semitones <= delta < 0:
        return int((2*num_semitones+1)+delta)
    else:
        #print(f'd2i: delta = {delta}, marking as a rest') #I'm assuming this is unlikely
        return int((2*num_semitones+1))


def index_2_delta(index, num_semitones=4):
    """turns indices back into deltas as described in delta_2_index"""

    if index <= num_semitones: #if delta is positive
        return int(index)
    elif num_semitones < index < 2*num_semitones+1: #if delta is negative 
        return int(index - (2*num_semitones+1))
    else:
        #print(f'i2d: index = {index}, marking as a rest')
        return int(100) #placeholder for rest
    

def get_delta_3d_lookup(train_dir, num_semitones=4, part='bass'):
    #returns the log probability matrix of P(delta | soprano_{n-1}, bass_n)
    #index by (bass, prev_sporano, delta)
    lookup_table = np.ones((128, 128, 2*num_semitones+2)) * 1/1000

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

            # add 1 to the lookup table at index (part, prev_soprano, delta)
            if part == 'bass':
                lookup_table[int(b), int(prev_s), delta_2_index(delta)] += 1
            elif part == 'alto':
                lookup_table[int(a), int(prev_s), delta_2_index(delta)] += 1
            elif part == 'tenor':
                lookup_table[int(t), int(prev_s), delta_2_index(delta)] += 1
            else:
                raise ValueError(f"Invalid part: {part}")

        # normalize the lookup table (for each delta)
        for i in range(lookup_table.shape[2]):
            lookup_table[:, :, i] = lookup_table[:, :, i] / np.sum(lookup_table[:, :, i])
            
        return np.log(lookup_table) #returns the log probability