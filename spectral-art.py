from PIL import Image, ImageDraw
from numpy.fft import rfft
from numpy import mean
import argparse
import wave
import struct
import sys

def verifyDecay(value):
    if float(value) < 0 or float(value) >= 1:
        raise argparse.ArgumentTypeError("Geometric signal decay value out of range (must be between 0 and 1)")
    return value

def readInSegment(f, startingPosition, runLength):
    audio_data = []
    f.rewind()
    f.setpos(f.getsampwidth() * f.getnchannels() * startingPosition)
    for i in range(0, runLength):
        waveData = waveFile.readframes(1)
        audio_data.append(mean(struct.unpack("<hh", waveData)))
    return audio_data

parser = argparse.ArgumentParser(description='Convert an audio segment to a art')
parser.add_argument('audio_file', type=str,
                    help='a file containing the sample audio data')
parser.add_argument('colour_file', type=open,
                    help='a file containing the colour codes')
parser.add_argument('output_file', type=str,
                    help='output file')
parser.add_argument('--begin', '-b', type=int, default=0,
                    help='time in seconds to start capture')
parser.add_argument('--end', '-e', default=60, type=int,
                    help='time in seconds to stop capture')
parser.add_argument('--height', '-y', default=2000, type=int,
                    help='height of the output image')
parser.add_argument('--width',  '-x', default=4000, type=int,
                    help='width of the output image')
parser.add_argument('--frequency-count', '-f', default=-1, type=int,
                    help='force a frequency count and linear interpolate the image')
parser.add_argument('--upper-frequency', '-u', default=20000, type=int,
                    help='upper frequency threshold for image (default: 20000Hz)')
parser.add_argument('--lower-frequency', '-l', default=20, type=int,
                    help='lower frequency threshold for image (default: 20Hz)')
parser.add_argument('--geometric-signal-decay', '-g', default=0, type=verifyDecay,
                    help='give data a decay method to smooth image (0-1)')
parser.add_argument('--scale-log', '-s', action='store_true',
                    help='scale frequency amplitude logarithmically')
parser.add_argument('--orient-log', '-o', action='store_true',
                    help='scale the location of the frequencies in the y axis logarithmically')
args = parser.parse_args()

waveFile = wave.open(args.audio_file, 'r')

sample_width  = waveFile.getsampwidth()
frame_rate    = waveFile.getframerate()
channel_count = waveFile.getnchannels()

print("Audio Sample Width:  {}".format(sample_width))
print("Audio Frame Rate:    {}".format(frame_rate))
print("Audio Channel Count: {}".format(channel_count))

im = Image.new('RGB', (args.width, args.height), color='red')
draw = ImageDraw.Draw(im)
seg = readInSegment(waveFile, 0, 2200)
print(rfft(seg))

# write to stdout
im.save("out.png", "PNG")
