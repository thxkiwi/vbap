import numpy as np

def mag2db(mag):
    return 20 * np.log10(mag)

def db2mag(db):
    return 10 ** (db / 20)

class Crossfader(object):
    def __init__(self, n):
        t = np.linspace(0, np.pi, n)
        cos_t = np.cos(t)
        self.multipliers = 0.5 * np.column_stack((1 + cos_t, 1 - cos_t))

    def crossfade(self, crossfade_from, crossfade_to):
        if crossfade_from.shape != crossfade_to.shape:
            raise ValueError('src and dst must have the same shape')
        
        if len(crossfade_from) < len(self.multipliers):
            multipliers = self.multipliers[:crossfade_from.shape[1],:]
            self.multipliers = self.multipliers[crossfade_from.shape[1],:]
            done = False
        else:
            # self.multipliers[-1,:] should == [0, 1]
            # Extend multipliers to the length needed using the last element of multipliers
            multipliers = np.vstack((self.multipliers, np.tile(self.multipliers[-1,:], (len(crossfade_from) - len(self.multipliers), 1))))
            done = True

        from_to = np.column_stack((crossfade_from, crossfade_to))
        m = multipliers * from_to
        s = np.sum(m, axis=1)
        return s, done

if __name__ == '__main__':
    rnd = np.random.default_rng()
    
    fs = 48000
    T = 60
    t = np.linspace(0, T, fs*T)

    x = np.sin(2 * np.pi * 440 * t)
    y = np.sin(2 * np.pi * 880 * t)

    crossfade_start = rnd.integers(fs*T)
    crossfade_end = rnd.integers(crossfade_start, fs*T)

    print(f'Crossfade [{crossfade_start}, {crossfade_end}) {crossfade_end - crossfade_start} samples; t=[{1.0*crossfade_start/fs}s, {1.0*crossfade_end/fs}s) {(float(crossfade_end) - crossfade_start)/fs}s')
    crossfader = Crossfader(crossfade_end - crossfade_start)

    output = np.zeros_like(x)
    output[:crossfade_start] = x[:crossfade_start]

    done = False
    block_start = crossfade_start
    while not done:
        block_size = rnd.integers(block_start, fs*T)
        print(f'block_size={block_size} samples; t={block_size/fs}s')
        output[block_start:block_start + block_size], done = \
            crossfader.crossfade(x[block_start:block_start + block_size], 
                                 y[block_start:block_start + block_size])
        block_start += block_size

    import scipy

    scipy.io.wavfile.write(f'crossfaded-{int(float(crossfade_start)/fs)}-{int(float(crossfade_end)/fs)}.wav', fs, output)
        
    pass
