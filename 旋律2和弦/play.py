'''
使用了music21这个非常好的python音乐库，将模型得到的四节《走马》和弦，结合主旋律播放

silence   2019.12.14
'''

from music21 import *
from 旋律2和弦.evaluate import play1,play2,play3,play4
from 旋律2和弦.get_chords import *

Repeat_Time=1
stream1 = stream.Stream()
stream1.insert(0.0,instrument.ElectricBass())


# chords=play1.split(' ')
# chords=chords[:-1]
# for i in range(4):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
# chords=sugar1.split(' ')
# chords=chords[:-1]
# for i in range(8):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
# chords=play2.split(' ')
# chords=chords[:-1]
# for i in range(4,8):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
#
# chords=play3.split(' ')
# chords=chords[:-1]
# for i in range(4):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
# chords=sugar2.split(' ')
# chords=chords[:-1]
# for i in range(8):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
# chords=play4.split(' ')
# chords=chords[:-1]
# for i in range(4,8):
#     stream1.repeatAppend(getchord(chords[i]),Repeat_Time)


#处理evaluate得到的四句和弦，进行相应的音乐转化
chords=play1.split(' ')
chords=chords[:-1]
for i in range(8):
    stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
    i= i + 2
chords=play2.split(' ')
chords=chords[:-1]
for i in range(8):
    stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
    i = i + 2
chords=play3.split(' ')
chords=chords[:-1]
for i in range(8):
    stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
    i = i + 2
chords=play4.split(' ')
chords=chords[:-1]
for i in range(8):
    stream1.repeatAppend(getchord(chords[i]),Repeat_Time)
    i = i + 2


score=stream.Score()
#和弦音部分
part1=stream.Part()
#旋律音部分
part2=stream.Part()


part1.append(stream1)
#这里的主旋律选择的是我很喜欢的一首音乐——《走马》
part2.append(converter.parse("tinyNotation: 4/4 c'8 a'8 a'8 a'8 a'4 g'8 g'4 d'8 d'8 e'8 d'2 d'8 d'8 d'8 d'8 d'4 a8 c'4 d'4 e'4 r8 d'8 c'8 c'8 a'8 a'8 a'8 a'4 g'8 g'4 d'4 d'4 r8 d'8 c'8 e'8 g'.4~ g'2 r2 r2 c'8 a'8 a'8 a'8 a'4 g'8 g'4 d'8 d'8 e'8 d'2 d'8 d'8 d'8 d'8 d'4 e'8 d'4 c'4 c'4 c'8 c'8 a8 c'8 d'.4 c'8 d'4 c'4 d'2 r8 a4 c'1 r1"))
#配入和弦音
score.insert(0,part1)
#配入主旋律
score.insert(0,part2)
#midi文件，一般视频播放器都可以直接播放
score.show("midi")