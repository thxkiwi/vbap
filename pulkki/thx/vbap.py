import numpy as np
import math
import thx.util

class PannedSource:
    """
    `PannedSource` class for panning a monaural source between two emitters 
    where the source contributes to each emitter by a given gain.

    Attributes:
    -----------
    source_angle : float
        Angle in degrees of the source in the horizontal plane.
    
    panning_pair : array-like
        Array-like of two integers that are the indices of the emitters in the 
        panning pair.
    
    gains : array-like
        Gains for each emitter in the pair.

    Methods:
    --------
    __str__():
        Returns a string representation of the panned source.
    
    pan(source_signal):
        Pans the given source signal between the two emitters in the panning pair.
        
        Parameters:
        -----------
        source_signal : array-like
            Source signal of `m` samples with shape (m, 1).
        
        Returns:
        --------
        np.ndarray
            An m x 2 array with the source panned between the two emitters in 
            the group.
    """

    # \todo: Add emitter angles
    def __init__(self, source_angle, panning_pair, gains):
        """
        Initialize the VBAP (Vector Base Amplitude Panning) object for a source 
        at the given angle to be panned between the two emitters in the panning 
        pair by the respective gains.

        Parameters:
        -----------
        source_angle : float
            Angle in degrees of the source in the horizontal plane.

        panning_pair : array-like
            Array-like of two integers that are the indices of the emitters that 
            index the system's VectorBasePanner emitter_angles array.

        gains : numpy.ndarray
            Gains for each emitter in the pair. Values below the machine epsilon 
            for float32 are set to 0.0.
        """

        self.source_angle = source_angle
        self.panning_pair = panning_pair
        self.gains = gains
        self.gains[gains < np.finfo(np.float32).eps] = 0.0
        

    def __str__(self):
        # \todo: Add emitter angles
        return (f'Source at {np.degrees(self.source_angle):7.2f}° := {self.gains[0]:0.4f} * '
                f'{self.panning_pair[0]:2} + '
                f'{self.gains[1]:0.4f} * {self.panning_pair[1]:2}')
    
    def pan(self, source_audio):
        """
        Pan the given source audio between two emitters.

        Parameters:
        -----------
        source : numpy.ndarray
            A 1-dimensional array of shape (m, 1) representing the audio source 
            to be panned. 'm' is the number of samples.

        Returns:
        --------
        numpy.ndarray
            A 2-dimensional array of shape (m, 2) where the source is panned 
            between the two emitters in the group. 
            
            Axes:
                0:  Time (samples).
                1:  Channel.
        """

        return self.gains * np.repeat(source_audio, 2).reshape(len(source_audio), 2)

class VectorBasePanner(object):
    """
    Class for Vector Base Amplitude Panning (VBAP).

    The VectorBasePanner class provides methods to initialize and perform vector 
    base amplitude panning using given emitter and source angles.
    
    Emmitter angles and source angles are given in degrees. The class calculates
    the gains for each source and emitter pair and pans the sources to the output
    channels.
    
    Emitter and source angles can be updated after initialization.

    Attributes:
    -----------
    MINUS_3dB : float
        A constant representing -3dB in magnitude.

    Methods:
    --------
    __init__(emitter_angles, source_angles):
        Initializes the VBAP class with emitter and source angles.

    __calculate_gains__():
        Calculates the gains for each source and emitter pair.

    __valid_panned_sources__():
        Returns a list of valid PannedSource objects with valid gains.

    set_source_angles(source_angles):
        Sets new source angles for the VBAP.

    pan(sources, output):
        Pans the given sources to the output channels.

    Raises:
    -------
    ValueError:
        If the number of rows in sources is not equal to the number of rows in 
        output.
    ValueError:
        If the number of sources is not equal to the number of sources in the 
        configuration.
    ValueError:
        If the maximum channel index in the configuration is greater than or 
        equal to the number of output channels.

    Returns:
    --------
    numpy.ndarray
        The output array with the sources panned to the output channels.
        
    See Also:
    ---------
    [Virtual Sound Source Positioning Using Vector Base Amplitude Panning - Pulkki AES 1997](https://thxlive.sharepoint.com/:b:/s/THXMobileSoftwareEngineering/EZtBJ6o-IbNOlz1OsNYhZOUB0E0JQEIT3FF3u3eLmRUO6Q?e=uEPdjj).
    """

    MINUS_3dB = thx.util.db2mag(-3.0)

    def __init__(self, emitter_angles, source_angles, normalize_gains=True):
        """
        Initialize the VBAP (Vector Base Amplitude Panning) class with emitter 
        and source angles.

        Args:
        -----
        emitter_angles : Array-like of tuples
            Each tuple contains an angle in degrees and the index of the emitter 
            in the output channels.

        source_angles : Array-like of floats
            Angles of the sound sources in degrees.
        """

        # The angles of the emitters in radians.
        self.__emitter_angles__ = None
        
        # Indices of the emitters in the output channels.
        self.__emitter_channel_indices__ = None
        
        # Unit vectors for the emitters. Note that the matrix is constructed as 
        # the transpose of the vectors. When P' = G * L, the result of P' is the 
        # correct p = g1 L1 + g2 L2 by scaled vector addition. The projections 
        # of the unit vectors sum to the unit vector in the direction of the 
        # source.
        self.__L__ = None
        
        # Pairs of indices into self.__emitter_angles__ such that each pair 
        # forms an active arc between which sources may be panned.
        # Create successive pairs of speaker angles until looping around. 
        # The pair (n1, n2) where n1 is 'more clock-wise' than n2.
        self.__panning_pairs__ = None
        
        # The angles of the sound sources in radians.
        self.__source_angles__ = None
        
        # The cartesian directional unit vectors for the sound sources.
        self.__P__ = None

        # G is the matrix that contains the calculated gains between each emitter 
        # within a panning pair of emitters, for each emitter.
        # A 3D array of shape (M, N, 2) containing the normalized gains for each
        # source and each emitter.
        self.__G__ = None
        
        self.__G_normalized__ = None
        self.__G_unnormalized__ = None
        
        self.normalize_gains = normalize_gains
        
        # A list of valid PannedSource objects with valid gains.
        self.__panned_sources__ = None
        
        self.__set_emitter_angles__(emitter_angles)
        self.__set_source_angles__(source_angles)
        self.__on_updated__()
        
        def __setattr__(setattrself, name, value):
            """
            Override the __setattr__ method to update when self.normalize_gains is set.

            Parameters:
            -----------
            name : str
                The name of the attribute to set.
            
            value : Any
                The value to set the attribute to.
            """
            setattrself.__super__.__setattr__(name, value)
            if name in ['normalize_gains']:
                if value:
                    setattrself.__G__ = setattrself.__G_normalized__
                else:
                    setattrself.__G__ = setattrself.__G_unnormalized__
                
        self.__setattr__ = __setattr__

        
    def __calculate_gains__(self):
        """
        Calculate the gains for each source and each emitter.

        This method computes the gains for M sound sources and N panning pairs,
        where each panning pair consists of two emitters forming an active arc. 
        The gains are normalized for each source and panning pair.

        Sets the gains in the self.__G__ attribute.

        Notes
        -----
        - The method assumes that `self.__source_angles__`, `self.__emitter_angles__`,
          `self.__P__`, `self.__panning_pairs__`, and `self.__L__` are already defined
          and properly initialized.
        - The gains are calculated using the inverse of the matrix formed by the
          panning pairs and the source positions.
        - The resulting gains are normalized to ensure they are unit vectors.
        """
        
        self.__G_unnormalized__ = np.zeros(
            (len(self.__source_angles__), len(self.__emitter_angles__), 2))

        self.__G_normalized__ = np.zeros(
            (len(self.__source_angles__), len(self.__emitter_angles__), 2))

        # Preallocate and reuse the matrix that contains the gains for each 
        # source and each emitter.
        Ln1n2 = np.zeros((2,2))

        # For M sound sources, for N panning pairs, [n1, n2] where n1 is 
        # 'more clock-wise' than n2.
        for m in range(len(self.__source_angles__)):
            # Must use reshape. numpy 1D arrays have shape (n,) and not (n, 1).
            pT = self.__P__[m].reshape(1, 2)
            for n in range(len(self.__panning_pairs__)):
                panning_pair = np.array(self.__panning_pairs__[n])
                Ln1n2[:,:] = self.__L__[panning_pair,:]
                self.__G_unnormalized__[m, n, :] = pT @ np.linalg.inv(Ln1n2)

                # Normalize the gains
                norm = np.linalg.norm(self.__G_unnormalized__[m, n, :])
                self.__G_normalized__[m, n, :] = self.__G_unnormalized__[m, n, :] / norm
                
        if self.normalize_gains:
            self.__G__ = self.__G_normalized__
        else:
            self.__G__ = self.__G_unnormalized__

    def __find_valid_panned_sources__(self):
        """
        A PannedSource is considered valid if:
        - The first gain is 0.0 and the second gain is 1.0 (when a sound source is
          at the exact angle of an emitter), or
        - All gains are strictly between 0 and 1.

        Returns:
            list: A list of ``PannedSource`` objects with valid gains.
        """
        
        # \todo: Add emitter angles
        self.__panned_sources__ = \
            [PannedSource(source_angle=self.__source_angles__[m], 
                          panning_pair=self.__panning_pairs__[n],
                          gains=self.__G__[m, n, :])
                for m in range(len(self.__source_angles__))
                for n in range(len(self.__emitter_angles__))
                                        
                # The source is entirely in the most counter-clockwise emitter...
                if ((self.__G__[m, n, 0] == 0.0) \
                    and (self.__G__[m, n, 1] == 1.0)) 
                                        
                # ...or the source is entirely between the emitters.
                or np.logical_and(self.__G__[m, n, :] > 0, 
                                self.__G__[m, n, :] < 1)
                    .all()
            ]
    
    def __set_emitter_angles__(self, emitter_angles):
        """
        Set the emitter angles and compute related attributes.
        This method sorts the provided emitter angles, converts them to 
        radians, and computes the corresponding directional unit vectors. 
        It also determines the panning pairs based on the sorted angles.
        
        Parameters
        ----------
        emitter_angles : array-like
            A list or array of emitter angles and their corresponding 
            channel indices. The array should have shape (N, 2), where N 
            is the number of emitters. The first column contains the angles 
            in degrees, and the second column contains the channel indices.
        
        Attributes
        ----------
        __emitter_angles__ : ndarray
            Sorted emitter angles in radians.
        
        __emitter_channel_indices__ : ndarray
            Channel indices corresponding to the sorted emitter angles.
        
        __L__ : ndarray
            Array of shape (N, 2) containing the cosine and sine of the 
            sorted emitter angles.
        
        __panning_pairs__ : list of tuples
            List of tuples representing pairs of indices for panning.
        """
        # Sort emitter_angles based on the angle.
        emitter_angles = np.array(emitter_angles)
        emitter_angles = emitter_angles[emitter_angles[:, 0].argsort()]
        
        self.__emitter_angles__ = np.radians(emitter_angles[:, 0])

        self.__emitter_channel_indices__ = emitter_angles[:, 1]

        self.__L__ = np.column_stack((np.cos(self.__emitter_angles__), 
                                      np.sin(self.__emitter_angles__)))
        
        # Works because self.__emitter_angles__ is sorted.
        # Successive pairs concatenated with the pair of the last and first.
        self.__panning_pairs__ = \
            [(i, i + 1) for i in range(len(self.__emitter_angles__) - 1)] \
                + [(len(self.__emitter_angles__) - 1, 0)]
        
    def __set_source_angles__(self, source_angles):
        """
        Set the source angles and calculate the corresponding cartesian 
        directional unit vectors.

        Parameters
        ----------
        source_angles : array-like
            Angles of the sound sources in degrees. These angles will be 
            converted to radians.

        Notes
        -----
        The method converts the provided source angles from degrees to 
        radians and then calculates the cartesian directional unit vectors 
        for the sound sources. The unit vectors are stored in the attribute 
        `self.__P__`.
        """
        self.__source_angles__ = np.radians(source_angles)

        # Calculate the cartesian directional unit vectors for the sound sources.
        self.__P__ = np.column_stack((np.cos(self.__source_angles__), np.sin(self.__source_angles__)))
        
    def __on_updated__(self):
        """
        Private method that is called when an update occurs.

        This method performs the following actions:
        1. Calculates the gains by calling `__calculate_gains__`.
        2. Identifies valid panned sources by calling 
           `__find_valid_panned_sources__`.

        Returns:
            None
        """
        self.__calculate_gains__()
        self.__find_valid_panned_sources__()        
    
    def set_emitter_angles(self, emitter_angles):
        """
        Set the angles for the emitters.

        Updates the angles of the emitters and recalculates the gains and 
        valid panned sources.

        Parameters:
        -----------
        emitter_angles : Array-like of tuples
            Each tuple contains an angle (in degrees) and the index of the 
            emitter in the output channels.

        Returns:
        --------
        None
        """
        self.__set_emitter_angles__(emitter_angles)
        self.__on_updated__()
        
    def set_source_angles(self, source_angles):
        """
        Set the angles for the audio sources.

        This method updates the angles for the audio sources and recalculates 
        the gains and valid panned sources.

        Parameters
        ----------
        source_angles : list of float
            A list of angles (in degrees) for the audio sources.

        Returns
        -------
        None
        """
        self.__set_source_angles__(source_angles)
        self.__on_updated__()
    
    def pan(self, source_signal, output):
        """
        Pan the given sound sources to the output channels.

        Parameters
        ----------
        source_signal : ndarray
            shape: (block_size, len(source_angles))
            Array of sound source signals.

        output : ndarray
            shape: (block_size, number of output channels). 
            
            Number of output channels >= number of emitters.
            
            max(emitter_channel_indices) < number of output channels.
            
            The output array to which the panned sources will be summed.
            
        The number of rows in sources must be equal to the number of rows 
        in output. 
        
        Returns
        -------
        output : ndarray
            The output array with the panned sources summed into the 
            appropriate channels based on the emitter channel indices.

        Raises
        ------
        ValueError
            If the number of rows in sources is not equal to the number of 
            rows in output.

            If the number of sources is not equal to the number of sources 
            in the configuration.

            If the maximum channel index in the configuration is greater 
            than or equal to the number of output channels.
        """

        if source_signal.shape[0] != output.shape[0]:
            raise ValueError(f'The number of rows in sources must be equal to the number of rows in output. {source_signal.shape=}, {output.shape=}')

        if source_signal.shape[1] != len(self.__source_angles__):
            raise ValueError(f'The number sources must be equal to the number of sources in the configuration. {source_signal.shape=}, {self.__source_angles__.shape=}')
        
        if self.__emitter_channel_indices__.max() >= output.shape[1]:
            raise ValueError(f'The maximum channel index in the configuration must be less than the number of output channels. {self.__emitter_channel_indices__}, {output.shape=}')

        # Preallocate the panned outputs.
        panned_outputs = np.zeros((output.shape[0], len(self.__emitter_angles__)))

        # Pan each source to the panned outputs.
        for panned_source_index, panned_source in enumerate(self.__panned_sources__):
            panned_outputs[:, panned_source.panning_pair] \
                += panned_source.pan(source_signal[:, panned_source_index])

        # Find the channel indices of only the channels that had audio summed into them from the panning.

        # Extract panning pairs and gains from panned sources
        panned_pairs_gains = np.array([list(zip(panned_source.panning_pair, 
                                                panned_source.gains)) 
                                                for panned_source in self.__panned_sources__])

        # Flatten to emitter indices and gains
        panned_source_emitter_indices = panned_pairs_gains[:,:,0].astype(int).flatten()
        # panned_source_gains = panned_pairs_gains[:,:,1].flatten()

        # Find the unique indices of the emitters that had audio summed into them.
        unique_panned_source_emitter_indices = np.unique(panned_source_emitter_indices)
        unique_panned_source_channel_indices = [self.__emitter_channel_indices__[emitter] for emitter in unique_panned_source_emitter_indices]

        # Sum the panned outputs into the output channels according to the self.emitter_channel_indices.
        for (emitter_index, channel_index) in enumerate(self.__emitter_channel_indices__):
            output[:, channel_index] += panned_outputs[:, emitter_index]

        # Subtract 3dB from the channels that had any panned audio summed into them.
        output[:, unique_panned_source_channel_indices] *= self.MINUS_3dB

        return output
