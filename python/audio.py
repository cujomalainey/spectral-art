import wave
import struct

def real(v):
    return v.real

class betterWav:
    def __init__(self, f, args):
        self.f = f
        self.input_length  = int(ceil(f.getframerate() * args.height / (args.upper_frequency - args.lower_frequency))) if args.frequency_count == -1 else args.frequency_count

        print("Audio Sample Width:    {}".format(f.getsampwidth()))
        print("Audio Frame Rate:      {}".format(f.getframerate()))
        print("Audio Channel Count:   {}".format(f.getnchannels()))
        print("Audio FFT Input Depth: {}".format(self.input_length))

    def readInSegment(self, startingTime):
        audio_data = []
        frame_start = self.getStartingPosition(self.getFrameFromTime(startingTime))
        self.f.setpos(self.frameToPointer(frame_start))
        for i in range(0, self.input_length):
            # TODO dynamically parse format
            audio_data.append(mean(struct.unpack("<hh", self.f.readframes(1))))
        return audio_data

    def frameToPointer(self, frame):
        return self.f.getsampwidth() * self.f.getnchannels() * frame

    def getFrameFromTime(self, time):
        return int(round(time * self.f.getframerate()))

    def getStartingPosition(self, startingPosition):
        startingPosition -= int(ceil(self.input_length / 2))
        startingPosition = max(startingPosition, 0)
        startingPosition = min(self.f.getnframes() - self.input_length, startingPosition)
        return startingPosition


waveFile = betterWav(wave.open(args.audio_file, 'r'), args)
