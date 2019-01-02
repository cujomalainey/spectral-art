from PIL import Image, ImageDraw
import argparse

def verifyDecay(value):
    if float(value) < 0 or float(value) >= 1:
        raise argparse.ArgumentTypeError("Geometric signal decay value out of range (must be between 0 and 1)")
    return value

parser = argparse.ArgumentParser(description='Convert an audio segment to a art')
parser.add_argument('colour_file', type=open,
                    help='a file containing the colour codes')
parser.add_argument('output_file', type=str,
                    help='output file')
parser.add_argument('--begin', '-b', type=int, default=0,
                    help='time in seconds to start capture')
parser.add_argument('--end', '-e', default=60, type=int,
                    help='time in seconds to stop capture')
parser.add_argument('--height', '-y', default=200, type=int,
                    help='height of the output image')
parser.add_argument('--width',  '-x', default=200, type=int,
                    help='height of the output image')
parser.add_argument('--frequency-count', '-f', default=-1, type=int,
                    help='force a frequency count and linear interpolate the image')
parser.add_argument('--upper-frequency', '-u', default=20000, type=int,
                    help='upper frequency threshold for image')
parser.add_argument('--lower-frequency', '-l', default=20, type=int,
                    help='lower frequency threshold for image')
parser.add_argument('--geometric-signal-decay', '-g', default=0, type=verifyDecay,
                    help='give data a decay method to smooth image (0-1)')
parser.add_argument('--scale-log', '-s', action='store_true',
                    help='scale frequency amplitude logarithmically')
parser.add_argument('--orient-log', '-o', action='store_true',
                    help='scale the location of the frequencies in the y axis logarithmically')
args = parser.parse_args()

im = Image.new('RGB', (128, 128), color='red')

draw = ImageDraw.Draw(im)
draw.line((0, 0) + im.size, fill=128)
draw.line((0, im.size[1], im.size[0], 0), fill=128)
del draw

# write to stdout
im.save("out.png", "PNG")
