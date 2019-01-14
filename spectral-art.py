from numpy.fft import rfft
from numpy import mean, array, amax
from math import ceil, log
from tqdm import tqdm
import argparse
import sys

def verifyDecay(value):
    if float(value) < 0 or float(value) >= 1:
        raise argparse.ArgumentTypeError("Geometric signal decay value out of range (must be between 0 and 1)")
    return float(value)


parser = argparse.ArgumentParser(description='Convert an audio segment to art')
parser.add_argument('audio_file', type=str,
                    help='a file containing the sample audio data')
parser.add_argument('colour_file', type=open,
                    help='a file containing the colour codes')
parser.add_argument('output_file', type=str,
                    help='output file')
parser.add_argument('--begin', '-b', type=float, default=0,
                    help='time in seconds to start capture')
parser.add_argument('--end', '-e', default=60, type=float,
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
parser.add_argument('--scale', '-s', choices=["linear", "root", "log"], default="root",
                    help='scale frequency amplitude logarithmically')
parser.add_argument('--orient-log', '-o', action='store_true',
                    help='scale the location of the frequencies in the y axis logarithmically')
parser.add_argument('--scale-percentile', '-p', default=0.95, type=float,
                    help='percentile which to clip scale of on data')
args = parser.parse_args()



time_delta = (args.end - args.begin) / args.width
current_time = args.begin

img_data = []

print("Analyzing audio...")
for i in tqdm(range(0, args.width)):
    seg = waveFile.readInSegment(current_time)
    img_data.append(list(map(lambda x: abs(real(x)), rfft(seg))))
    current_time += time_delta

for i in range(1, len(img_data)):
    for x in range(0, len(img_data[i])):
        img_data[i][x] = img_data[i - 1][x] * args.geometric_signal_decay + img_data[i][x] * (1 - args.geometric_signal_decay)

temp_data = array(img_data).flatten()
temp_data.sort()
max_val = temp_data[int(len(temp_data)*args.scale_percentile)]
print("Scaling to: {}".format(max_val))
print("Drawing image...")
img_data = [list(map(lambda y: min(y / max_val, 1), x)) for x in img_data]

for x in range(0, args.width):
    for y in range(0, args.height):
        scale = int(255*img_data[x][y])
        draw.point((x, y), fill=(scale, scale, scale))
im.save(args.output_file, "PNG")
