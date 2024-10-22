import numpy as np
import math
import util

class PannedSource:
    def __init__(self, source_angle, panning_pair, gains):
        '''
        source_angle: angle in degrees of the source in the horizontal plane.

        panning_pair: array-like of two integers that are the indices of the emitters in the panning pair.

        gains: gains for each emitter in the pair.
        '''
        self.source_angle = source_angle
        self.panning_pair = panning_pair
        self.gains = gains
        self.gains[gains < np.finfo(np.float32).eps] = 0.0
        

    def __str__(self):
        return f'{self.source_angle} = {self.gains[0]} * {self.panning_pair[0]} + {self.gains[1]} * {self.panning_pair[1]}'
    
    def pan(self, source):
        '''
        source.shape == (m, 1)
        input.shape == (m, n) where m is the number of samples and n is the number of channels/speakers.

        Returns an m x 2 array with source panned between the two emitters in the group.
        '''

        return self.gains * np.repeat(source, 2).reshape(len(source), 2)

class VectorBasePan:

    # Set class variables
    MINUS_3dB = util.db2mag(-3.0)


    def __init__(self, emitter_angles, source_angles):
        '''
        emitter_angles: list of tuples angles in degrees of the emitters in the horizontal plane, and their index in the output channel configurtion.

        source_angles: list of angles in degrees of the sources in the horizontal plane.

        n_output_channels: number of output channels in the configuration. Allows for creation of the output channels that do not receive any panned sound, i.e., center in 5.1 and 7.1.

        The VBAP algorithm is based on the paper "A Vector Base Amplitude Panning Method for Spatial Reproduction" by Ville Pulkki.

        The algorithm calculates the gains for each emitter for each source, such that the sound source is perceived to be in the direction of the source.
        '''

        # Sort emitter_angles based on the angle.
        emitter_angles = np.array(emitter_angles)
        emitter_angles = emitter_angles[emitter_angles[:, 0].argsort()]
        
        # Stored emitter_angles are just the angles.
        self.emitter_angles = np.radians(emitter_angles[:, 0])

        self.emitter_channel_indices = emitter_angles[:, 1]

        # Unit vectors for the emitters. Not that the matrix is constructed as the transpose of the vectors.
        # When P' = G * L, the result of P' is the correct p = g1 L1 + g2 L2 by scaled vector addition.
        # The projections of the unit vectors sum to the unit vector in the direction of the source.
        self.L = np.column_stack((np.cos(self.emitter_angles), np.sin(self.emitter_angles)))
        
        # Create successive pairs of speaker angles until looping around. The pair (n1, n2) where 
        # n1 is 'more clock-wise' than n2.
        self.panning_pairs = [(i, i + 1) for i in range(len(self.emitter_angles) - 1)] + [(len(self.emitter_angles) - 1, 0)]

        self.source_angles = np.radians(source_angles)

        # Calculate the cartesian directional unit vectors for the sound sources.
        self.P = np.column_stack((np.cos(self.source_angles), np.sin(self.source_angles)))

        # G is the matrix that contains the calculated gains between each emitter within
        # a panning pair of emitters, for each emitter.
        #
        # For M sound sources, for N panning pairs, [n1, n2] where n1 is 'more clock-wise' than n2.
        self.G = np.zeros((len(self.source_angles), len(self.emitter_angles), 2))

    def gains(self):

        # Preallocate and reuse the matrix that contains the gains for each source and each emitter.
        Ln1n2 = np.zeros((2,2))

        for m in range(len(self.soruce_angles)):
            # Must use reshape. numpy 1D arrays have shape (n,) and not (n, 1).
            pT = self.P[m].reshape(1, 2)
            for n in range(len(self.panning_pairs)):
                panning_pair = np.array(self.panning_pairs[n])
                Ln1n2[:,:] = self.L[panning_pair,:]
                self.G[m, n, :] = pT @ np.linalg.inv(Ln1n2)

                # Normalize the gains
                self.G[m, n, :] /= np.linalg.norm(self.G[m, n, :])

        return self.G

    def panned_sources(self):
        valid_pannings = np.logical_and(self.G > 0, self.G < 1.00001)
        return [PannedSource(source_angle=self.source_angles[m],
                            panning_pair=self.panning_pairs[n],
                            gains=self.G[m, n, :]) 
                            for m in range(len(self.source_angles)) 
                            for n in range(len(self.emitter_angles)) 
                            if valid_pannings[m,n,:].all()]
    
    def pan(self, sources, output):
        '''
        sources: (blockSize x len(source_angles)) array of sound sources.

        output: (blockSize x n) array of output channels.

            self.emitter_channel_indices < n

        '''
        if sources.shape[0] != output.shape[0]:
            raise ValueError('The number of rows in sources must be equal to the number of rows in output.')

        if sources.shape[1] != len(self.source_angles):
            raise ValueError('The number of columns in sources must be equal to the number of sources in the configuration.')
        
        if self.emitter_channel_indices.max() >= output.shape[1]:
            raise ValueError('The maximum channel index in the configuration must be less than the number of output channels.')

        # Preallocate the panned outputs.
        panned_outputs = np.zeros((output.shape[0], len(self.panned_sources)))

        # Pan each source to the panned outputs.
        panned_sources = self.panned_sources()
        for m, panned_source in enumerate(panned_sources):
            panned_outputs[:, panned_source.panning_pair] += panned_source.pan(sources[:, m])

        # # Subtract 3dB from the panned outputs because theey were summed.
        # panned_outputs *= self.MINUS_3dB

        # # Divisors to normalize the panned outputs.
        # divisors = np.abs(panned_outputs).max(axis=1)

        # # Avoid division by zero.
        # divisors[divisors < np.finfo(np.float32).eps] = 1.0
        
        # # Normalize the panned outputs.
        # panned_outputs /= divisors.reshape(len(divisors), 1)

        # Sum the panned outputs into the output channels according to the self.emitter_channel_indices.
        for i in range(len(self.emitter_channel_indices)):
            output[:, self.emitter_channel_indices[i]] += panned_outputs[:, i]

        return output
        
if __name__ == "__main__":
    dir(np)