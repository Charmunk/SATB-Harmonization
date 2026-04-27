import midi_parser as mp
import numpy as np
import importlib
import matplotlib.pyplot as plt
import os

def getChordToS(dataset_dir, numChords=21, numNotes=128, debug=False):
    """
    Construct pdf for soprano line given chord
    Inputs:
        - dataset_dir: directory specifying midis of interest
        - numChords: number of possible chords (int), default 24... should be equal to len(chord_roll)
        - numNotes: number of possible notes (int), default 128... should be equal to len(S[0])
        - debug: if True, print debug messages
    Outputs:
        - output = [chord_to_s, prevchord_to_s]: list of matrices
            - chord_to_s: a (numNotes) x (numChords) matrix, each entry specifying P(S|C)
            - prevchord_to_s: also (numNotes) x (numChords), but P(S|C_t-1)
    """
    chord_to_s = np.zeros((numNotes, numChords), dtype=float)
    prevchord_to_s = np.zeros((numNotes, numChords), dtype=float)
    output = [chord_to_s, prevchord_to_s]
    cwd = os.getcwd()
    attempts = 0
    fails = 0
    for song_folder in os.listdir(dataset_dir):
        if song_folder != ".DS_Store":
            midi_filepath = os.path.join(cwd, dataset_dir, song_folder, "mix")
            attempts += 1
            try:
                C, S = mp.get_chord_roll_and_S(midi_filepath)
                sampDur = len(S[0])
                for i in range(len(output)):
                    transmat = output[i]
                    # Get P(S|chord)
                    for samp in range(i, sampDur):
                        samp_notes = S[:,samp]
                        samp_chords = C[:, samp-i] # if i=1, get previous sample chord
                        transmat = np.add(transmat, np.outer(samp_notes, samp_chords), dtype=float)                    
                    # Get P(S|chord_t-1)
                    norm = np.sum(transmat, axis=0)
                    norm_mat = 0.000001*np.ones(np.shape(transmat))
                    norm_mat = np.add(norm_mat, (np.matlib.repmat(norm, len(transmat), 1)))
                    output[i] += transmat/norm_mat
            except:
                if debug:
                    print(f"couldn't parse file {midi_filepath}")
                fails += 1
    if debug:
        print(f"successfully processed {(attempts-fails)/attempts}% of files")
    return output

if __name__ == "__main__":
    import midi_parser as mp
    import numpy as np
    import importlib
    import matplotlib.pyplot as plt
    import os
    importlib.reload(mp)
    
    dataset_dir = "dataset/1/train"
    
    [chord_to_s, prevchord_to_s] = getChordToS(dataset_dir, debug=False)
    matching_elems = np.sum(np.isclose(prevchord_to_s, chord_to_s))
    total_elems = np.shape(chord_to_s)[0]*np.shape(chord_to_s)[1]
    print(f"P(S|chord) is different from P(S|prevchord): {not(matching_elems == total_elems)}")
    print(f"total matching elements: {matching_elems}")
    print(f"total elements: {total_elems}")
    print(f"total distinct elements: {np.abs(total_elems - matching_elems)}")
