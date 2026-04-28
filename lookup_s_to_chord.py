'''
Turns a sequence of soprano notes into a sequence of chords.

Input:
    - sequence of soprano notes
Output:
    - sequence of chords

Algorithm:
    P(c_t | s_t, c_t-1) = P(s_t | c_t) * P(c_t | c_t-1) / P(s_t | c_t-1)
'''

import midi_parser as mp
import numpy as np
import importlib
import matplotlib.pyplot as plt
import os
importlib.reload(mp)
from chord_to_s import getChordToS
from chord_to_chord_mm import chord_to_chord_mm

def s_to_chord(dataset_dir, numChords=21, numNotes=128, debug=False):
    '''
    Turns a sequence of soprano notes into a sequence of chords.

    Input:
        - sequence of soprano notes
        - chord_to_s: a (numNotes) x (numChords) matrix, each entry specifying P(S|C)
    Output:
        - sequence of chords
    '''
    chord_to_s, prevchord_to_s = getChordToS(dataset_dir, numChords=21, numNotes=128, debug=False)
    prevchord_to_chord = chord_to_chord_mm(dataset_dir)
    
    #element-wise multiplication and division
    s_to_chord_matrix = chord_to_s * prevchord_to_chord / prevchord_to_s

    return s_to_chord_matrix