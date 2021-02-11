import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import entropy, median_abs_deviation
from scipy.ndimage import median_filter
from tqdm.auto import tqdm
tqdm.pandas()


GRAVITY_CUTOFF_HZ = 0.5
NOISE_CUTOFF_HZ = 20
MAXVAL, MINVAL = 6, -6  # Note: GENEActiv has a range of +/-6g whereas Axivity and Actigraph +/-8 or more


class Processing():
    def __init__(self, **kwargs):
        self.sampleRate = kwargs['sampleRate']
        self.epochPeriod = kwargs['epochPeriod']
        self.epochFile = kwargs['epochFile']
        self.extractor = None
        if kwargs.get('extractFeatures', True):
            self.extractor = FeatureExtractor(
                sampleRate=kwargs.get('sampleRate'), 
                epochPeriod=kwargs.get('epochPeriod')
            )


    def run(self, npyFile):
        data = Processing.loadNpyToFrame(npyFile)
        #TODO quality checks
        data = Processing.regularizeSampleRate(data, self.sampleRate)
        data = Processing.filterNoise(data, self.sampleRate)
        #TODO
        # data = calibrateGravity(...)
        #TODO
        # data = detectNonwear(...)

        if self.extractor is not None:
            epochFeats = self.extractor.run(data)
            epochFeats.to_csv(self.epochFile)


    @staticmethod
    def loadNpyToFrame(npyFile):
        data = pd.DataFrame(np.load(npyFile))
        data['time'] = data['time'].astype('datetime64[ms]')
        data.set_index('time', inplace=True)
        return data


    @staticmethod
    def regularizeSampleRate(data, sampleRate, method='nearest'):
        samplePeriodNanos = int(1000_000_000/sampleRate)  # in nanos
        if method == 'nearest':
            return data.resample(f'{samplePeriodNanos}N').nearest(limit=1)
        elif method == 'linear':
            raise NotImplementedError
        else:
            raise ValueError


    @staticmethod
    def filterNoise(data, sampleRate):
        xyzCols = ['x', 'y', 'z']

        # Temporarily fill nan values
        mask = data[xyzCols].isna().any(1)
        data[xyzCols] = data[xyzCols].fillna(method='ffill')

        # Clip unrealistically high values
        data[xyzCols] = np.clip(data[xyzCols].values, MINVAL, MAXVAL)
        # Noise removal by median filtering
        data[xyzCols] = median_filter(data[xyzCols].values, (5,1), mode='nearest')
        # Noise removal by lowpass filtering
        data[xyzCols] = butterfilt(data[xyzCols].values, NOISE_CUTOFF_HZ, sampleRate, axis=0)

        # Restore NaN values
        data.loc[mask, xyzCols] = np.nan

        return data


    #TODO
    @staticmethod
    def calibrateGravity():
        pass


    #TODO
    @staticmethod
    def detectNonwear():
        pass


class FeatureExtractor():
    def __init__(self, sampleRate, epochPeriod) -> None:
        self.sampleRate = sampleRate
        self.epochPeriod = epochPeriod


    def run(self, data):
        epochs = data.groupby(pd.Grouper(freq=f'{self.epochPeriod}S'))

        featsFrame = {}
        for t, epoch in tqdm(epochs):
            xyz = epoch[['x', 'y', 'z']].values

            # Check if good chunk else return NaNs
            if (np.isfinite(xyz).all() and (len(xyz) == self.epochPeriod * self.sampleRate)):
                feats = FeatureExtractor.extract(xyz, self.sampleRate)
            else:
                feats = {name:np.nan for name in FeatureExtractor.featureNames()}

            featsFrame[t] = feats

        featsFrame = pd.DataFrame.from_dict(featsFrame, orient='index')

        return featsFrame


    @staticmethod
    def extract(xyz, sampleRate):
        return {
            **FeatureExtractor.basicFeatures(xyz, sampleRate),
            # **FeatureExtractor.sanDiegoFeatures(xyz, sampleRate),
            # **FeatureExtractor.unileverFeatures(xyz, sampleRate),
        }


    @staticmethod
    def featureNames():
        return [
            *FeatureExtractor.basicFeaturesNames(),
            *FeatureExtractor.sanDiegoFeaturesNames(),
            *FeatureExtractor.unileverFeaturesNames()
        ]

    
    @staticmethod
    def basicFeatures(xyz, sampleRate):
        feats = {}
        feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
        feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
        feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)

        x, y, z = xyz.T

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['xyCorr'] = np.nan_to_num(np.corrcoef(x, y)[0,1])
            feats['yzCorr'] = np.nan_to_num(np.corrcoef(y, z)[0,1])
            feats['zxCorr'] = np.nan_to_num(np.corrcoef(z, x)[0,1])

        m = np.linalg.norm(xyz, axis=1)
        feats['enmoTrunc'] = np.mean(np.maximum(m - 1.0, 0.0))
        feats['mad'] = stats.median_abs_deviation(m)

        return feats


    @staticmethod
    def basicFeaturesNames():
        return ['xMean', 'yMean', 'zMean',
                'xStd', 'yStd', 'zStd',
                'xRange', 'yRange', 'zRange',
                'xyCorr', 'yzCorr', 'zxCorr',
                'enmoTrunc', 'mad']


    @staticmethod
    def sanDiegoFeatures(xyz, sampleRate):
        """ Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification 
        https://pubmed.ncbi.nlm.nih.gov/26673126/ """

        feats = {}

        # Body stream
        xyzb = butterfilt(xyz, (GRAVITY_CUTOFF_HZ, NOISE_CUTOFF_HZ), sampleRate, axis=0)
        xb, yb, zb = xyzb.T
        mb = np.linalg.norm(xyzb, axis=1)

        # Gravity stream
        xyzg = butterfilt(xyz, GRAVITY_CUTOFF_HZ, sampleRate, axis=0)
        xg, yg, zg = xyzg.T

        feats['sanMean'], feats['sanStd'] = np.mean(mb), np.std(mb)
        feats['sanCoefVar'] = feats['sanMean'] / (feats['sanStd'] + 1e-8)
        feats['sanMin'], feats['sanMax'] = np.min(mb), np.max(mb)
        feats['san25p'], feats['sanMedian'], feats['san75p'] = np.percentile(mb, (25, 50, 75))
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['sanAutocorr'] = np.nan_to_num(np.corrcoef(mb[sampleRate:], mb[:-sampleRate])[0,1])

        # Roll, pitch, yaw
        yaw, roll, pitch = np.arctan2(xb, yb), np.arctan2(yb, zb), np.arctan2(zb, xb)
        feats['sanYawAvg'], feats['sanRollAvg'], feats['sanPitchAvg'] = np.mean(yaw), np.mean(roll), np.mean(pitch)
        feats['sanYawStd'], feats['sanRollStd'], feats['sanPitchStd'] = np.std(yaw), np.std(roll), np.std(pitch)
        # Roll, pitch, yaw using gravity stream
        yawg, rollg, pitchg = np.arctan2(xg, yg), np.arctan2(yg, zg), np.arctan2(zg, xg)
        feats['sanYawgAvg'], feats['sanRollgAvg'], feats['sanPitchgAvg'] = np.mean(yawg), np.mean(rollg), np.mean(pitchg)

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['sanxyCorr'] = np.nan_to_num(np.corrcoef(xb,yb)[0,1])
            feats['sanyzCorr'] = np.nan_to_num(np.corrcoef(yb,zb)[0,1])
            feats['sanzxCorr'] = np.nan_to_num(np.corrcoef(zb,xb)[0,1])

        m = np.linalg.norm(xyz, axis=1)

        # Spectrum using Welch's method with 5s segment length
        freqs, powers = welch(m, fs=sampleRate, nperseg=5*sampleRate, detrend='constant')

        feats['sanEntropy'] = entropy(powers)

        # Dominant frequency
        idxmax = np.argmax(powers)
        feats['sanDominantFreq'], feats['sanDominantPower'] = freqs[idxmax], powers[idxmax]
        # Dominant frequency between 0.3-3Hz
        mask = (0.3 <= freqs) & (freqs <= 3)
        idxmax = np.argmax(powers[mask])
        feats['sanDominantFreq_0.3_3'], feats['sanDominantPower_0.3_3'] = freqs[mask][idxmax], powers[mask][idxmax]

        # Powers for 1Hz, 2Hz, ..., 15Hz using Welch's method with 1s segment length
        _, powers15 = welch(m, fs=sampleRate, nperseg=sampleRate, detrend='constant')
        powers15 = powers15[1:16]
        feats.update({f'sanFFT{i+1}':val for i, val in enumerate(powers15)})

        return feats


    @staticmethod
    def sanDiegoFeaturesNames():
        return ['sanMean', 'sanStd', 'sanCoefVar',
                'sanMin', 'sanMax', 'san25p', 'sanMedian', 'san75p', 
                'sanAutocorr',
                'sanYawAvg', 'sanRollAvg', 'sanPitchAvg', 
                'sanYawStd', 'sanRollStd', 'sanPitchStd', 
                'sanYawgAvg', 'sanRollgAvg', 'sanPitchgAvg',
                'sanEntropy', 
                'sanDominantFreq', 'sanDominantPower', 
                'sanDominantFreq_0.3_3', 'sanDominantPower_0.3_3',
                *[f'sanFFT{i+1}' for i in range(15)],
                ]


    @staticmethod
    def unileverFeatures(xyz, sampleRate):
        """ Physical Activity Classification Using the GENEA Wrist-Worn Accelerometer
        https://pubmed.ncbi.nlm.nih.gov/21988935/
        """
        feats = {}
        m = np.linalg.norm(xyz, axis=1)

        # Spectrum using Welch's method with 5s segment length
        freqs, powers = welch(m, fs=sampleRate, nperseg=5*sampleRate, detrend='constant')

        # Dominant and 2nd dominant between 0.3-15Hz
        mask = (0.3 <= freqs) & (freqs <= 15)
        _freqs, _powers = freqs[mask], powers[mask]
        idxs = np.argsort(_powers)
        feats['dominantFreq_0.3_15'], feats['dominantPower_0.3_15'] = _freqs[idxs[-1]], _powers[idxs[-1]]
        feats['secondDominantFreq_0.3_15'], feats['secondDominantPower_0.3_15'] = _freqs[idxs[-2]], _powers[idxs[-2]]

        # Total power between 0.3-15Hz
        feats['totalPower_0.3_15'] = np.sum(_powers)

        # Dominant between 0.6-2.5Hz
        mask = (0.6 <= freqs) & (freqs <= 2.5)
        _freqs, _powers = freqs[mask], powers[mask]
        idx = np.argmax(_powers)
        feats['dominantFreq_0.6_2.5'], feats['dominantPower_0.6_2.5'] = _freqs[idx], _powers[idx]

        return feats


    @staticmethod
    def unileverFeaturesNames():
        return ['dominantFreq_0.3_15', 'dominantPower_0.3_15',
                'secondDominantFreq_0.3_15', 'secondDominantPower_0.3_15',
                'totalPower_0.3_15',
                'dominantFreq_0.6_2.5', 'dominantPower_0.6_2.5']


def butterfilt(x, cutoffs, fs, order=4, axis=0):
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        btype = 'bandpass'
        low, high = cutoffs
        Wn = (low / nyq, high / nyq)
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = butter(order, Wn, btype=btype, analog=False, output='sos')
    y = sosfiltfilt(sos, x, axis=axis)
    return y

