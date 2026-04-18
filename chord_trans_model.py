def getChordTransMat(chord_roll_list, C=24):
    """ 
    Inputs: 
        - chord_roll_list: a list of arrays, where each array specifies the chord played in each (quarter note time frame) for a single training recording
        - C: possible chords
    Outputs:
        - chord_trans_mat: the (CxC chord transition probability matrix
    """

    chord_trans_mat_raw = np.zeros((C, C), dtype=int)

    # for each array of states (in each frame for a single training recording)
    # look at each state in the array (state), and the state it transitions to in the array (nextstate)
    # and increment the value at A_raw[state, nextstate] to encode that state transition
    for chord_roll in chord_roll_list:
      for i in range(C-1):
        chord = chord_roll[i]
        nextchord = chord_roll[i+1]
        chord_trans_mat_raw[chord, nextchord] += 1

    # normalize by row so the sum of each state transition probability for each state is 1
    norm = np.sum(chord_trans_mat_raw, axis=1)
    norm_mat = np.transpose(np.matlib.repmat(norm,len(chord_trans_mat_raw),1))
    chord_trans_mat = chord_trans_mat_raw/norm_mat
    return chord_trans_mat

if __name__ == "__main__":
   import numpy as np
   chord_roll_list = []
   getChordTransMat(chord_roll_list)