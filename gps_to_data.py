import glob
import guitarpro
import hashlib
import struct
import os

for gp3_path in glob.iglob('data/**/*.gp?', recursive=True):
    print(gp3_path)
    try:
        tab = guitarpro.parse(gp3_path)
    except Exception:
        continue

    for i, track in enumerate(tab.tracks):
        if track.isPercussionTrack:
            continue
        if track.is12StringedGuitarTrack:
            continue
        if track.isBanjoTrack:
            continue
        if len(track.strings) != 6:
            continue
        if tuple((s.number, s.value) for s in track.strings) != ((1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40)):
            continue

        h = hashlib.sha256(f"{gp3_path}track{i}".encode('utf-8')).hexdigest()
        
        file_empty = True

        with open(f"train_data/{h}", 'wb') as f:
            for measure in track.measures:
                signature = measure.header.timeSignature
                if signature.numerator != 4:
                    continue
                if signature.denominator.value != 4:
                    continue
                measure_parts = [[-1,-1,-1,-1,-1,-1]]*16
                for beat in measure.voices[0].beats:
                    sixteenth_start = (beat.start - measure.start)//240
                    sixteenth_duration = beat.duration.time//240
                    while sixteenth_duration > 0:
                        for note in beat.notes:
                            if sixteenth_start + sixteenth_duration - 1 > 15:
                                continue
                            if note.string-1 > 5:
                                continue
                            measure_parts[sixteenth_start + sixteenth_duration - 1][note.string-1] = note.value
                        sixteenth_duration -= 1
                for measure_part in measure_parts:
                    f.write(struct.pack("bbbbbb", *measure_part))
                    file_empty = False
                    #print(measure_part)
                #print("---")

        if file_empty:
            os.remove(f"train_data/{h}")

