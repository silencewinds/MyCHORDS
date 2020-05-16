'''
为了配合music21的python音乐库，将翻译得到的和弦（字符形式）转化为可调用的和弦（音乐形式）

silence   2019.12.14
'''

from music21 import *
LENGTH=2
F = chord.Chord(['F3', 'A3', 'C4', 'F4'])
F.quarterLength = LENGTH
G = chord.Chord(['G3', 'B3', 'D4', 'G4'])
G.quarterLength = LENGTH
Em = chord.Chord(['E3', 'G3', 'B3', 'E4'])
Em.quarterLength = LENGTH
Dm = chord.Chord(['D3', 'A3', 'C4', 'F4'])
Dm.quarterLength = LENGTH
C = chord.Chord(['C3', 'E3', 'G3', 'C4'])
C.quarterLength = LENGTH
C7 = chord.Chord(['C3', 'E3', 'G3', 'Bb3'])
C7.quarterLength = LENGTH
Am = chord.Chord(['A3', 'C4', 'E4', 'A4'])
Am.quarterLength = LENGTH
E = chord.Chord(['E3', 'G#3', 'B3', 'E4'])
E.quarterLength = LENGTH
Dm7 = chord.Chord(['D3', 'F3', 'A3', 'C4'])
Dm7.quarterLength = LENGTH
G7 = chord.Chord(['G3', 'B3', 'D4', 'F4'])
G7.quarterLength = LENGTH
G5 = chord.Chord(['G3', 'B3', 'D3', 'G4'])
G5.quarterLength = LENGTH
Am7 = chord.Chord(['A3', 'C4', 'E3', 'G4'])
Am7.quarterLength = LENGTH
Dm = chord.Chord(['D3', 'F3', 'A3', 'D4'])
Dm.quarterLength = LENGTH
FM7 = chord.Chord(['F3', 'A3', 'C4', 'E4'])
FM7.quarterLength = LENGTH
Em7 = chord.Chord(['E3', 'G3', 'B3', 'D4'])
Em7.quarterLength = LENGTH
CM7 = chord.Chord(['C3', 'E3', 'G3', 'B3'])
CM7.quarterLength = LENGTH
A = chord.Chord(['A3', 'C#4', 'E4', 'A4'])
A.quarterLength = LENGTH
D7 = chord.Chord(['D3', 'F#3', 'A3', 'C4'])
D7.quarterLength = LENGTH
Am6 = chord.Chord(['A3', 'C4', 'E4', 'F#4'])
Am6.quarterLength = LENGTH
Fm = chord.Chord(['F3', 'Ab3', 'C4', 'F4'])
Fm.quarterLength = LENGTH
GM7 = chord.Chord(['G3', 'B3', 'D4', 'F#4'])
GM7.quarterLength = LENGTH
F7 = chord.Chord(['F3', 'A3', 'C4', 'Eb4'])
F7.quarterLength = LENGTH
Fmaj = chord.Chord(['F3', 'A3', 'C4', 'F4'])
Fmaj.quarterLength = LENGTH
Gm7 = chord.Chord(['G3', 'Bb3', 'D4', 'F4'])
Gm7.quarterLength = LENGTH
Fdim = chord.Chord(['F3', 'Ab3', 'B3', 'F4'])
Fdim.quarterLength = LENGTH
E7 = chord.Chord(['E3', 'G#3', 'B3', 'D4'])
E7.quarterLength = LENGTH
A7 = chord.Chord(['A3', 'C#4', 'E4', 'G4'])
A7.quarterLength = LENGTH
Bb6 = chord.Chord(['Bb3', 'C4', 'E3', 'G4'])
Bb6.quarterLength = LENGTH
Fm7 = chord.Chord(['F3', 'Ab3', 'C4', 'Eb4'])
Fm7.quarterLength = LENGTH
Am7 = chord.Chord(['A3', 'C4', 'E3', 'G4'])
Am7.quarterLength = LENGTH
D = chord.Chord(['D3', 'F#3', 'A3', 'D4'])
D.quarterLength = LENGTH
Am7 = chord.Chord(['A3', 'C4', 'E3', 'G4'])
Am7.quarterLength = LENGTH
Bm7 = chord.Chord(['B3', 'D4', 'F#4', 'A4'])
Bm7.quarterLength = LENGTH
G6 = chord.Chord(['G3', 'B3', 'D4', 'E4'])
G6.quarterLength = LENGTH

def getchord(char):
    dic={"F":F,"G":G,"Em":Em,"Am":Am,"Dm":Dm,"C":C,"C7":C7,"Em":Em,"E":E,"Dm7":Dm7,"G7":G7,"G5":G5,"Am7":Am7,"Dm":Dm,"FM7":FM7,"Em7":Em7,"CM7":CM7,"A":A,
         "D7":D7,"Am6":Am6,"Fm":Fm,"C7":C7,"GM7":GM7,"F7":F7,"Fmaj":Fmaj,"Gm7":Gm7,"Fdim":Fdim,"E7":E7,"A7":A7,"Bb6":Bb6,"Fm7":Fm7,"D":D,"Bm7":Bm7,"G6":G6}
    return dic.get(char)
