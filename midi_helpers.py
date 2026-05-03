import mido
import wave
from pathlib import Path

#if you want to output to a midi file
def arrays_to_midi(tracks_tuple, output_filename="output.mid", bpm=120):
    """
    Converts a tuple of 1D arrays (MIDI notes) into a MIDI file.
    
    Args:
        tracks_tuple: Tuple of 1D arrays/lists containing MIDI note numbers.
                      Use 0 or None for silence/rests.
        output_filename: Name of the file to save.
        bpm: Beats per minute.
    """
    # 4 PPQ means each array index is a 16th note (in 4/4 time)
    ppq = 4 
    mid = mido.MidiFile(ticks_per_beat=ppq)
    
    # Calculate microseconds per beat for the tempo track
    tempo = mido.bpm2tempo(bpm)
    
    for track_data in tracks_tuple:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Optional: Add tempo to the first track
        if mid.tracks.index(track) == 0:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        last_note = None
        ticks_since_last_event = 0

        for note in track_data:
            # Case 1: Note change or Note ending
            if note != last_note:
                # If a note was playing, turn it off
                if last_note is not None and last_note > 0:
                    track.append(mido.Message('note_off', note=int(last_note), 
                                         velocity=64, time=ticks_since_last_event))
                    ticks_since_last_event = 0
                
                # If the new value is a note, turn it on
                if note is not None and note > 0:
                    track.append(mido.Message('note_on', note=int(note), 
                                         velocity=64, time=ticks_since_last_event))
                    ticks_since_last_event = 0
                
                last_note = note
            
            # Increment time (1 tick per array index)
            ticks_since_last_event += 1

        # Clean up: Close the last note if the array ends while playing
        if last_note is not None and last_note > 0:
            track.append(mido.Message('note_off', note=int(last_note), 
                                 velocity=64, time=ticks_since_last_event))

    mid.save(output_filename)
    print(f"Successfully saved to {output_filename}")

# if you want to output to a wav file and play it in jupyter
import numpy as np
import IPython.display as ipd

def midi_arrays_to_ipd(tracks_tuple, bpm=120, sample_rate=44100, save_wav=False, wav_filename="output.wav"):
    """
    Synthesizes MIDI arrays where repeated consecutive notes are held 
    as a single continuous tone.
    """
    step_duration = (60.0 / bpm) / 4
    num_steps = max(len(t) for t in tracks_tuple)
    
    total_samples = int(num_steps * step_duration * sample_rate)
    master_buffer = np.zeros(total_samples)

    def midi_to_freq(note):
        if note is None or note <= 0: return 0
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    for track in tracks_tuple:
        i = 0
        while i < len(track):
            note = track[i]
            freq = midi_to_freq(note)
            
            if freq > 0:
                # 1. Figure out how long this note is held
                start_step = i
                while i < len(track) and track[i] == note:
                    i += 1
                end_step = i
                
                # 2. Calculate the continuous duration
                hold_count = end_step - start_step
                note_duration = hold_count * step_duration
                
                # 3. Synthesize the long "held" wave
                t_held = np.linspace(0, note_duration, int(note_duration * sample_rate), False)
                tone = np.sin(2 * np.pi * freq * t_held)
                
                # 4. Apply envelope only at the start and end of the hold
                fade_len = min(int(sample_rate * 0.005), len(tone) // 2) 
                envelope = np.ones_like(tone)
                if fade_len > 0:
                    envelope[:fade_len] = np.linspace(0, 1, fade_len)
                    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
                
                # 5. Mix into master buffer
                start_idx = int(start_step * step_duration * sample_rate)
                end_idx = start_idx + len(tone)
                
                # Ensure we don't overshoot the buffer due to rounding
                master_buffer[start_idx:end_idx] += (tone * envelope * 0.2)
            else:
                # It's a rest, just move to the next index
                i += 1

    peak = np.max(np.abs(master_buffer))
    if peak > 0:
        audio_norm = master_buffer / peak
    else:
        audio_norm = master_buffer

    if save_wav:
        wav_path = Path(wav_filename)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(audio_norm, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
    return ipd.Audio(audio_norm, rate=sample_rate)
