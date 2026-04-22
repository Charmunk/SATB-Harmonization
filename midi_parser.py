import mido
from mido import MidiFile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pychord


def unpack_midi(midi_filename):
    """
    arguments: path to a midi file (str or Path). If the value has no suffix,
        it is interpreted as a filename under the local 'midi files/' folder
        (e.g. unpack_midi('mix') loads 'midi files/mix.mid').
    returns: four matrices S, A, T, B of dimension (num_pitches) x (num quarter notes) containing note info for each part
    #also return a vector for each of base chord (e.g. B not B7) over time
    """
    
    # path = midi_path
    # path = glob.glob(midi_path)
    # path = Path(midi_path)
    # if path.suffix == '':
    #     path = Path(path.name + '.mid')
    #     # path = Path.PureWindowsPath(path.name + '.mid')
    #     print(path)
    # mid = MidiFile(path)

    # load in the file
    mid = MidiFile(Path('midi files', midi_filename+'.mid'))


    meta_track = mid.tracks[0]
    numerator, denominator = 4, 4
    tempo = 500000
    for msg in meta_track:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            numerator = msg.numerator
            denominator = msg.denominator

    PPQ = mid.ticks_per_beat

    length_sec = mid.length

    length_ticks = mido.second2tick(length_sec, PPQ, tempo)

    num_beats = int(length_ticks / PPQ)

    metadata = [tempo, PPQ, length_ticks, (numerator, denominator)]
    # assumes will be returned in SATB order
    list_of_parts = []

    for i in range(1, len(mid.tracks)):
        track = mid.tracks[i]

        # frequency is hardcoded to 4 times per quarter note
        list_of_parts.append(get_piano_roll(track, PPQ, 4, num_beats))

    return list_of_parts, metadata


def get_piano_roll(track, PPQ, freq, num_beats, plot=False):
    total_samples = freq * num_beats
    output_array = np.zeros((128, total_samples))

    # Track the "start time" of currently active notes
    # stores {note_number: start_sample_index}
    open_notes = {}
    absolute_tick = 0

    for msg in track:
        absolute_tick += msg.time

        current_sample = int((absolute_tick * freq) / PPQ)

        is_note_on = (msg.type == 'note_on' and msg.velocity > 0)
        is_note_off = (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))

        if is_note_on:
            # If the note was already "on", close it first (prevents stuck notes)
            if msg.note not in open_notes:
                open_notes[msg.note] = current_sample

        elif is_note_off:
            if msg.note in open_notes:
                start_sample = open_notes.pop(msg.note)
                end_sample = current_sample

                if end_sample > start_sample:
                    output_array[msg.note, start_sample:end_sample] = 1

        if current_sample >= total_samples:
            break

    # if a note was never turned off, extend it to the end
    for note, start_sample in open_notes.items():
        output_array[note, start_sample:] = 1

    if plot:
        plt.figure(figsize=(12, 6))
        plt.imshow(output_array, aspect='auto', origin='lower', cmap='gray_r')
        plt.xlabel('Time (Samples)')
        plt.ylabel('MIDI Pitch')
        plt.show()

    return output_array


def index_to_note(midi_index, roots=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
    names = roots
    return f"{names[midi_index % 12]}"

def rearrange_chord(chord, root):
    """takes a base note and chord and returns the notes sorted in ascending order"""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    positions = pychord.analyzer.notes_to_positions(chord, root)
    positions = [position % 12 for position in positions] #change to normalize by base note
    root_index = names.index(root)
    output_chord = []

    for i in range(len(positions)):
        position = min(positions) #grab the smallest position (closest to base note)
        positions.remove(position) #remove it
        output_chord.append(names[(root_index+position) % 12])
    return output_chord


def roll_to_chord(roll):
    """ arguments: takes a piano roll matrix 
    returns a tuple of chords and indices
    """

    pitches, samples = roll.shape
    output = []

    for i in range(samples):
        indices = np.nonzero(roll[:, i])[0] #this returns a tuple of an array, the first thing is the stuff we want

        notes = []
        for j in range(len(indices)): #for each midi index, turn that into a note
            notes.append(index_to_note(indices[j]))

        #turn the notes into a chord
        ordered_notes = rearrange_chord(notes, notes[0]) #assume the bass note is the bottom note
        chord = pychord.analyzer.find_chords_from_notes(ordered_notes) #can have it just return the notes

        #transpose the chord so it's in C major
        if chord != []:
            # print(chord)
            if len(chord) == 1:
                # if isinstance(chord, str): # already a root name
                #     output.append((i, chord[0])) #return the chord with its timestamp
                # else:
                #     output.append((i, chord[0].root))
                output.append((i, chord[0].root))
                # output.append((i, 9999))


            else: #if there are multiple suggested chords
                output.append((i, chord[0].root)) #just take the first guess? could improve with circle of fifths knowledge
                # output.append((i, chord[0][0].root)) #just take the first guess? could improve with circle of fifths knowledge

    return output


def midi_parse(midi_path):
    """
    End-to-end pipeline that turns a MIDI file into a list of chord-name strings
    suitable for the markov model in chord_to_chord_mm.py.

    arguments:
        midi_path: path (str or Path) to a .mid file, e.g. 'dataset/1/song001/mix.mid'
    returns:
        list of chord strings (one per 16th-note sample where a chord was
        identified), e.g. ['D', 'D', 'Bm', 'G', 'Cmaj7', ...]

    TODO: collapse runs of identical chords, transpose to C major using the
    song's key signature, and restrict quality to major/minor.
    """
    parts, _metadata = unpack_midi(midi_path)
    roll = sum(parts)

    chord_tuples = roll_to_chord(roll)

    chord_strings = []
    for _sample_idx, chord in chord_tuples:
        if isinstance(chord, str):
            # multi-guess fallback from roll_to_chord: already just a root name
            chord_strings.append(chord)
        else:
            # list of pychord.Chord objects; take the first one as its string form
            chord_strings.append(str(chord[0]))

    return chord_strings


def getRootToIdxDict(roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
    """
    Input:
        - roots: list specifying potential chord roots, with each root represented as a string
                      (e.g. 'A' or 'C#)
        - C: number of unique chords in key signatures
    Output:
        - root_to_idx: dictionary mapping chord root to its index
    """
    root_to_idx = {r: i for i, r in enumerate(roots)}
    # alias enharmonic spellings (flats + theoretical E#/B#/Cb/Fb) to a canonical index
    enharmonic_aliases = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
        'E#': 'F',  'B#': 'C',  'Cb': 'B',  'Fb': 'E',
    }
    for alias, canonical in enharmonic_aliases.items():
        root_to_idx[alias] = root_to_idx[canonical]
    return root_to_idx


def getChordRoll(chords, sampDur, chordToIdxDict=dict):
    """
    Inputs:
        - chords: List (sampDur x 2) of tuples of the form (sample, chord_root) (with dtypes (int,str))
        - sampDur: duration of original audio, in samples (typically quarter notes)
        - chordToIdxDict: dictionary mapping {chord_identifier:index(int)}
    Outputs:
        - chord_roll: (C x sampDur) matrix specifying the activated chord(s) at each time sample
    """
    C = len(chordToIdxDict)
    # sampDur = chords[-1][0] + 1
    chord_roll = np.zeros((C, sampDur))

    for (samp, chord) in chords:
        chord_idx = chordToIdxDict[chord]
        try:
            chord_roll[chord_idx, samp] = 1
        except:
            print("Chord construction failed!")
            print(f"chord index: {chord_idx}")
            print(f"sample number: {samp}")


    return chord_roll
    
def get_chord_roll_and_S(midi_path):
    """
    End-to-end pipeline that turns a MIDI file into chord_roll and soprano line piano_roll 

    arguments:
        midi_path: path (str or Path) to a .mid file, e.g. 'dataset/1/song001/mix.mid'
    returns:
        chord_roll: list of chord strings (one per 16th-note sample where a chord was
                    identified), e.g. ['D', 'D', 'Bm', 'G', 'Cmaj7', ...]
        S: matrix specifying piano roll for just soprano part
    """
    [S, A, T, B], _metadata = unpack_midi(midi_path)
    roll = S+A+T+B
    sampDur = len(S)
    chord_strs = roll_to_chord(roll)
    chord_to_idx_dict = getRootToIdxDict()
    chord_roll = getChordRoll(chord_strs, sampDur, chord_to_idx_dict)
    return chord_roll, S

if __name__ ==  "__main__":
    import numpy as np
    import glob
    print("hello")