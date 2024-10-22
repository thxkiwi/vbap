import numpy as np

def mag2db(mag):
    return 20 * np.log10(mag)

def db2mag(db):
    return 10 ** (db / 20)

