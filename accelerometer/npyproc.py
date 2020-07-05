import pytz
from datetime import datetime
import functools
import os
import time
import random
import math
import argparse
from scipy import interpolate
from scipy.signal import resample, decimate
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pickle
import gzip
import blosc
from numba import jit, guvectorize
import jpype
import jpype.imports
import catch22
from catch22 import catch22_all
from tqdm.auto import tqdm

# in centiseconds
SEC = 100
MIN = 60 * SEC
HOUR = 60 * MIN
DAY = 24 * HOUR

# in nanoseconds
MICROS_IN_NANOS = 1000
MILLIS_IN_NANOS = 1000 * MICROS_IN_NANOS
CENTIS_IN_NANOS = 10 * MILLIS_IN_NANOS
SEC_IN_NANOS = 1000 * MILLIS_IN_NANOS
MIN_IN_NANOS = 60 * SEC_IN_NANOS
HOUR_IN_NANOS = 60 * MIN_IN_NANOS
DAY_IN_NANOS = 24 * HOUR_IN_NANOS

# in seconds
MIN_IN_SECS = 60
HOUR_IN_SECS = 60 * MIN_IN_SECS

# data format
TIME_FIELD = 'time'
X_FIELD, Y_FIELD, Z_FIELD = 'x', 'y', 'z'
TIME_DTYPE = 'i8'
XYZ_DTYPE = 'f4'
DATA_DTYPE = np.dtype([('time', 'i8'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
DEVICE_SAMPLE_HZ = 100

class Extractor():
    ''' A wrapper of the Java class FeatureExtractor using the JPype package.
    It starts a Java Virtual Machine, instantiates FeatureExtractor, and
    implements 'extract' method to handle numpy arrays.
    '''
    def __init__(self, basic=True, sanDiego=True, mad=True, unilever=True, fft3d=True):
        Extractor.start()
        self.java_extractor = jpype.JClass('FeatureExtractor')
        self.basic = basic
        self.sanDiego = sanDiego
        self.mad = mad
        self.unilever = unilever
        self.fft3d = fft3d

    def extract(self, xyz):
        xyz = xyz.astype('f8')
        xArray, yArray, zArray = xyz.T
        xArray = jpype.JArray(jpype.JDouble, 1)(xArray)
        yArray = jpype.JArray(jpype.JDouble, 1)(yArray)
        zArray = jpype.JArray(jpype.JDouble, 1)(zArray)
        return np.asarray(self.java_extractor.extract(
            xArray, yArray, zArray, DEVICE_SAMPLE_HZ,
            self.basic, self.sanDiego, self.mad, self.unilever, self.fft3d
        ))

    @staticmethod
    def shutdown():
    # Shut down Java Virtual Machine
        jpype.shutdownJVM()

    @staticmethod
    def start():
    # Start Java Virtual Machine
        if not jpype.isJVMStarted():
            jpype.addClassPath("java_feature_extractor")
            jpype.addClassPath("java_feature_extractor/JTransforms-3.1-with-dependencies.jar")
            jpype.startJVM("-XX:ParallelGCThreads=1", convertStrings=False)


# def quantile_features(xyz):
#     q = np.arange(.1, 1, .1)
#     v = np.linalg.norm(xyz, axis=1)
#     vxy = np.linalg.norm(xyz[:,[0,1]], axis=1)
#     vyz = np.linalg.norm(xyz[:,[1,2]], axis=1)
#     vzx = np.linalg.norm(xyz[:,[2,1]], axis=1)
#     vq = np.quantile(v, q)
#     vxyq = np.quantile(vxy, q)
#     vyzq = np.quantile(vyz, q)
#     vzxq = np.quantile(vzx, q)
#     return (vq, vxyq, vyzq, vzxq)


def quantile_features(xyz):
    vxy = np.linalg.norm(xyz[:,[0,1]], axis=1)
    vzx = np.linalg.norm(xyz[:,[2,1]], axis=1)
    vxyq10 = np.quantile(vxy, 0.1)
    vzxq30 = np.quantile(vzx, 0.3)
    vxyq60 = np.quantile(vxy, 0.6)
    return [vxyq10, vzxq30, vxyq60]


def catch22_features(xyz):
    xyz = xyz.astype('f8')
    v = np.linalg.norm(xyz, axis=1)
    vxy = np.linalg.norm(xyz[:,[0,1]], axis=1)
    x, y, z = xyz.T
    v, x, y, z = list(v), list(x), list(y), list(z)
    vxy = list(vxy)
    feats = []
    feats.append(catch22.SB_MotifThree_quantile_hh(v))
    feats.append(catch22.MD_hrv_classic_pnn40(y))
    feats.append(catch22.PD_PeriodicityWang_th0_01(x))
    feats.append(catch22.PD_PeriodicityWang_th0_01(v))
    feats.append(catch22.DN_OutlierInclude_p_001_mdrmd(v))
    feats.append(catch22.MD_hrv_classic_pnn40(vxy))
    feats.append(catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(v))
    return feats


def timer(msg):
    """Print runtime of the decorated function"""
    def inner(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            before = time.perf_counter()
            print(msg)
            value = func(*args, **kwargs)
            after = time.perf_counter()
            runtime = after - before
            print(f"done {func.__name__!r} ({runtime:.4f} secs)")
            return value
        return wrapper_timer
    return inner


@timer(msg="blosc-compressing... ")
def blosc_compress(x, clevel=9, shuffle=2, codec='lz4hc'):
    shuffle = (blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE)[shuffle]
    return blosc.compress_ptr(
        x.__array_interface__['data'][0], x.size, x.dtype.itemsize,
        clevel=clevel, shuffle=shuffle, cname=codec
    )


@timer(msg="blosc-decompressing... ")
def blosc_decompress(x, size, dtype, shape=None):
    x_ = np.empty(size, dtype=dtype)
    blosc.decompress_ptr(x, x_.__array_interface__['data'][0])
    return x_.reshape(shape)


def load_data(npypath, *args, **kwargs):
    if npypath.endswith('.npy.gz'):
        data = np.load(gzip.GzipFile(npypath, 'rb'), *args, **kwargs)
    elif npypath.endswith('.npy'):
        data = np.load(npypath, *args, **kwargs)
    else:
        raise ValueError('unrecognized file')
    assert data.dtype == DATA_DTYPE, 'wrong data dtype'
    return data


def nanos2secs(x):
    return x / SEC_IN_NANOS


def secs2nanos(x):
    return x * SEC_IN_NANOS


def nanos2centis(x):
    return x / CENTIS_IN_NANOS


def centis2secs(x):
    return x / SEC


def secs2centis(x):
    return x * SEC


def round_secs2centis(t):
    t = secs2centis(t)
    t = np.rint(np.diff(t, prepend=0)).astype(np.int)
    t = np.cumsum(t)
    return t


def round_nanos2centis(t):
    t = nanos2centis(t)
    t = np.rint(np.diff(t, prepend=0)).astype(np.int)
    assert np.all(t[1:] >= 1), "Found rounding error in time"
    t = np.cumsum(t)
    return t


def floor_day(x):
    return (x // DAY) * DAY


def where_nan(xyz):
    nans = np.isnan(xyz).reshape(xyz.shape[0],-1).any(axis=-1)
    locs, counts = count_true(nans)
    return locs, counts


def where_notnan(xyz):
    notnans = np.logical_not(np.isnan(xyz).reshape(xyz.shape[0],-1).any(axis=-1))
    locs, counts = count_true(notnans)
    return locs, counts


@guvectorize(["void(float32[:], int64, float32[:])"], "(n),()->(n)")
def rolling_std(x, window, x_std):
    asum = 0.0
    asqrsum = 0.0
    count = 0
    for i in range(window):
        asum += x[i]
        asqrsum += x[i]**2
        count += 1
        mu = asum / count
        var = max(0, asqrsum/count - mu**2)
        x_std[i] = math.sqrt(var)

    for i in range(window, len(x)):
        asum += x[i] - x[i-window]
        asqrsum += x[i]**2 - x[i-window]**2
        mu = asum / count
        var = max(0, asqrsum/count - mu**2)
        x_std[i] = math.sqrt(var)


@jit(nopython=True)
def count_true(conditions):
    """
    input: [0,1,0,0,1,1,1,0,1]
    output: [1,4,8], [1,3,1]
    """
    locs = []
    counts = []
    switch = False
    count = 0
    for i, c in enumerate(conditions):
        if c:
            if switch:
                count += 1
            else:
                switch = True
                loc = i
                count = 1
        else:
            if switch:
                switch = False
                locs.append(loc)
                counts.append(count)
                count = 0
    if switch:  # remaining count
        locs.append(loc)
        counts.append(count)
    return locs, counts


def resolve_daylight_saving_time(t, zone='Europe/London'):
    """
    Important: t expresses time in nanoseconds but is NOT the Unix time. The
    value of t[0] is such that utcfromtimestamp(t[0]) represents a local time
    (not UTC!), and t[i>0] are time increments from that point forward.
    Because of this, for any other instant i>0 ahead, utcfromtimestamp(t[i])
    may no longer represent the local time due to possible DST crossover. For
    example, suppose utcfromtimestamp(t[0]) is meant to represent the local
    time in London and utcfromtimestamp(t[0]) is 2017-03-26 00:59:59, then
    for some instant i>0 ahead utcfromtimestamp(t[i]) might return 2017-03-26
    01:05:00 which does not exist since clocks in London jumped an hour
    forward that date at exactly 01:00:00. The issue is that
    utcfromtimestamp, as the name suggests, is meant to only work on the Unix
    timestamp, which is the number of (nano)seconds since a fixed point in
    time, namely 1970-1-1 00:00:00 UTC. Therefore the Unix timestamp is
    unambiguous and does not depend on timezones. t can be interpreted as a
    "localized Unix time", with the caveat that it does not take into account
    DST time changes.
    """

    t = t.copy()
    mask = np.ones_like(t, dtype=bool)
    offset = 0

    #* In the following, _utc_transition_times contains the crossover datetimes
    #* in local time for the given zone. To find the crossover in t we can
    #* convert all t[i] using utcfromtimestamp, then the first few t[i] will
    #* indeed represent local time right up to the crossover. An alternative is
    #* to convert the local datetimes in _utc_transition_times to "localized
    #* Unix time" (see function doc) by assuming that the local datetimes
    #* are UTC time. The latter is more efficient if t is very large.
    tz = pytz.timezone(zone)
    # transition times for the timezone, in Unix time (nanoseconds)
    transition_times = [
        secs2nanos(_.replace(tzinfo=pytz.utc).timestamp())
        for _ in tz._utc_transition_times[1:]
    ]
    t0 = tz.localize(datetime.utcfromtimestamp(nanos2secs(t[0])))
    t1 = tz.localize(datetime.utcfromtimestamp(nanos2secs(t[-1])))
    #! Minor bug: t0.dst() and t1.dst() assumes pushbacks happening at 1am instead of 2am
    #! This leads to errors if t1 is between 1am and 2am
    #! Only instance found in 100k UKBB participants
    if t0.dst() != t1.dst():
        offset = 1 if t1.dst() > t0.dst() else -1
        if offset == -1:
            # pytz stores dst crossover at 1am, but pushbacks happen at 2am local
            transition_times = np.asarray(transition_times) + HOUR_IN_NANOS
        for tt in transition_times:
            idx = np.searchsorted(t, tt)
            if 0 < idx < len(t):
                t[idx:] += offset * HOUR_IN_NANOS  #! inplace
                if offset == -1:
                    # when pushback, there is an hour overlap
                    jdx = np.searchsorted(t, tt - HOUR_IN_NANOS)
                    mask = np.zeros_like(t, dtype=bool)
                    mask[:jdx] = True
                    mask[idx:] = True
                break  # assuming there's at most one crossover in t
        print("found daylight saving time crossover:")
        print(*(datetime.utcfromtimestamp(nanos2secs(t[i])) for i in range(idx-1, idx+1)))

    print(f"device first measurement at: {datetime.utcfromtimestamp(nanos2secs(t[0]))} (local time)")
    print(f"device last measurement at: {datetime.utcfromtimestamp(nanos2secs(t[-1]))} (local time)")

    return t, mask, offset


@timer(msg="loading data... ")
def load_xyz(npypath, start_of_day, num_days=7, fill_value=np.nan):
    data = load_data(npypath)
    # data['time'] = data['time'] + (170)*24*60*60*SEC_IN_NANOS  #! debug  (check DST October)
    # data['time'] = data['time'] - (40)*24*60*60*SEC_IN_NANOS  #! debug  (check DST March)
    t, mask, offset = resolve_daylight_saving_time(data['time'])
    t = t[mask]
    t = round_nanos2centis(t)
    start_date = floor_day(t[0]).astype(np.int)
    # end_date = floor_day(t[-1]).astype(np.int)
    end_date = start_date + num_days * DAY
    tidxs = t - start_date  # use the centiseconds as indexes, starting from start_date
    n = end_date - start_date + DAY
    xyz = np.full((n, 3), fill_value=fill_value, dtype=XYZ_DTYPE)
    xyz[tidxs] = repack_fields(
        data[[X_FIELD, Y_FIELD, Z_FIELD]][mask]).view(XYZ_DTYPE).reshape(len(tidxs), -1)
    xyz = xyz[start_of_day:-(DAY-start_of_day)]  # trim ends at start of day
    start_time, end_time = start_date + start_of_day, end_date + start_of_day
    # ndays = (end_time - start_time) // DAY
    return start_time, end_time, offset, xyz


@timer(msg="loading data... ")
def load_xyz2(npypath, fill_value=np.nan):
    data = load_data(npypath)
    # t, mask, offset = resolve_daylight_saving_time(data['time'])
    # t = t[mask]
    offset = 0
    t = data['time']
    t = round_nanos2centis(t)
    n = t[-1] - t[0] + 1
    tidx = t - t[0]
    xyz = np.full((n, 3), fill_value=fill_value, dtype=XYZ_DTYPE)
    xyz[tidx] = repack_fields(
        data[[X_FIELD, Y_FIELD, Z_FIELD]]).view(XYZ_DTYPE).reshape(-1, 3)
    return t, xyz, offset


@timer(msg="flagging nonwear periods... ")
def flag_nonwear(
    xyz, fill_value=np.nan,
    nonwear_patience=HOUR,
    nonwear_std=13.0/1000,
    std_window=10*SEC,
    inplace=True
):

    notnan_locs, notnan_counts = where_notnan(xyz)
    edges = [(i, i+n) for i,n in zip(notnan_locs, notnan_counts)]
    nonwear_locs = []
    nonwear_counts = []
    for i,j in edges:
        x = xyz[:,0][i:j]
        y = xyz[:,1][i:j]
        z = xyz[:,2][i:j]
        x_std = np.empty_like(x)
        y_std = np.empty_like(y)
        z_std = np.empty_like(z)
        rolling_std(x, std_window, x_std)
        rolling_std(y, std_window, y_std)
        rolling_std(z, std_window, z_std)
        mask = (x_std < nonwear_std) & (y_std < nonwear_std) & (z_std < nonwear_std)
        _locs, counts = count_true(mask)
        _nonwear_locs_counts = [
            (_loc, count) for _loc, count in zip(_locs, counts) if count > nonwear_patience
        ]
        nonwear_locs.extend(i + _loc for _loc,_ in _nonwear_locs_counts)
        nonwear_counts.extend(count for _,count in _nonwear_locs_counts)

    if inplace:
        for loc, count in zip(nonwear_locs, nonwear_counts):
            xyz[:,0][loc:loc+count] = fill_value
            xyz[:,1][loc:loc+count] = fill_value
            xyz[:,2][loc:loc+count] = fill_value

    return nonwear_locs, nonwear_counts


@timer(msg="imputing nan values (interrupts, nonwear, etc.)... ")
def impute_nan(xyzs, inplace=True):
    """
    Vertically impute a list of rows by random segment selection. xyzs is a
    list_like of rows/arrays of equal length, e.g.:

    [[0, 1, 2, nan, 3],
     [1, nan, nan, 4, 7],
     [3, 4, 5, 5, nan]]

    Contiguous nan elements are jointly replaced, e.g. in the second row
    [nan,nan] is replaced by [1,2] or [4,5].

    Each element in the row/array can be multidimensional, e.g.

    [[[0, 2, 5], [3, 1, 1], [2, nan, 3]],
     [[1, nan, 7], [6, 1, 9], [3, 4, 7]],
     [[3, 2, 5], [3, 1, 4], [5, 5, nan]]]

    In this case, the whole element is replaced if it contains at least one
    nan, e.g. [1,nan,7] is replaced by [0,2,5] or [3,2,5].
    """
    chunks = []
    imputed_locs = []
    imputed_counts = []
    for i, xyz in enumerate(xyzs):
        locs, counts = where_nan(xyz)
        choices = list(range(len(xyzs)))
        choices.remove(i)
        for j in choices:
            _xyz = xyzs[j]
            dones = []
            for k, (loc, count) in enumerate(zip(locs, counts)):
                if not np.any(np.isnan(_xyz[loc:loc+count])):
                    imputed_locs.append((i, loc))
                    imputed_counts.append(count)
                    chunks.append(_xyz[loc:loc+count])
                    dones.append(k)
            # update locs & counts to exclude already imputed
            locs = np.delete(locs, dones)
            counts = np.delete(counts, dones)

    if inplace:
        for (i, loc), count, chunk in zip(imputed_locs, imputed_counts, chunks):
            xyzs[i][loc:loc+count] = chunk

    return imputed_locs, imputed_counts


@timer(msg="fft resampling... ")
def resample_xyz(xyz, num):
    xyz[-1] = np.copy(xyz[0])  # ensure periodicity
    x = resample(xyz[:,0], num=num)
    y = resample(xyz[:,1], num=num)
    z = resample(xyz[:,2], num=num)
    xyz = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    return xyz


@timer(msg="decimating... ")
def decimate_xyz(xyz, q):
    x = decimate(xyz[:,0], q=q)
    y = decimate(xyz[:,1], q=q)
    z = decimate(xyz[:,2], q=q)
    xyz = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    return xyz


@timer(msg="linearly interpolating... ")
def interp1d_xyz(xyz, q):
    t = np.arange(len(xyz))
    fx = interpolate.interp1d(t, xyz[:,0])
    fy = interpolate.interp1d(t, xyz[:,1])
    fz = interpolate.interp1d(t, xyz[:,2])
    tnew = np.arange(0, len(t), q)
    x = fx(tnew)
    y = fy(tnew)
    z = fz(tnew)
    xyz = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    return xyz


def random_intervals(xyz, length=3000, size=1000):
    # Choose intervals between 00 and 06
    idxs0 = np.random.choice(range(0, 6*HOUR // length), size=int(0.1*size))
    # Choose intervals between 06 and 24
    idxs1 = np.random.choice(range(6*HOUR // length, 24*HOUR // length), size=int(0.9*size))
    idxs = np.concatenate((idxs0, idxs1))
    m = 14*HOUR // length  # how many intervals from 10am till midnight
    n = 24*HOUR // length  # total number of intervals
    idxs = np.mod(idxs + m, n)  # roll so that 10am is 0th index
    days = np.random.randint(7, size=len(idxs))

    xyz = xyz.reshape(7,-1,3000,3)
    intervals = xyz[days, idxs]
    return intervals


def mad(xyz, n=3000, lognormalize=True):
    # empirical values based on 1k subjects
    LOG_MEAN = -4.2166924
    LOG_STD = 1.5712029
    SLEEP_CUTOFF = 0.0061321235
    LOG_SLEEP_CUTOFF = -5.094214177927501
    xyz = xyz.reshape(-1,n,xyz.shape[-1])
    v = np.linalg.norm(xyz, axis=-1)
    res = v - np.mean(v, -1, keepdims=True)
    res = np.abs(res).mean(-1)
    if lognormalize:
        res = np.log(res+1e-8)
        res = (res - LOG_MEAN) / LOG_STD
    return res


def impute_nan2(x):
    '''
    Impute a 2D array in the vertical direction by random substitution, e.g

    Input:
    [[0,   1,   2, nan,   3],
     [1, nan, nan,   4,   7],
     [3,   4,   5,   5, nan]]

    Output:  (one of many possibilities)
    [[0,   1,   2,   5,   3],
     [1,   1,   2,   4,   7],
     [3,   4,   5,   5,   7]]

    The imputation is greedy in the sense that each row is used to fill in
    the imputed row as much as possible before jumping to the next row.
    '''
    x_new = x.copy()
    n = x.shape[0]
    for i in range(n):
        xi = x_new[i]
        js = np.random.permutation(n)
        for j in js:
            if j == i: continue
            mask = np.any(~np.isfinite(xi), axis=-1)
            if not mask.any(): break
            xj = x[j]
            xi[mask] = xj[mask]

    return x_new


def process(
    npypath, impute=False, detect_nonwear=False,
    start_of_day=10*HOUR, num_days=7,
    resample_method='fft', resample_hz=None,
    compress=True, outfile='tmp',
    transpose=True,
    seed=42
):

    # imputation has randomness
    random.seed(seed)
    np.random.seed(seed)

    start_time, end_time, offset, xyz = load_xyz(npypath, start_of_day, num_days=num_days, fill_value=np.nan)

    if detect_nonwear:
        nonwear_locs, nonwear_counts = flag_nonwear(xyz, inplace=True, fill_value=np.nan)
        nonwear_times = [i + start_time for i in nonwear_locs]
        # nonwear_times = [datetime.utcfromtimestamp(centis2secs(i + start_time) for i in nonwear_locs]
        nonwear_times_durations = list(zip(nonwear_times, nonwear_counts))
    else:
        nonwear_times_durations = []

    if impute:
        xyzs = xyz.reshape(num_days, DAY, -1)  # group by days
        imputed_locs, imputed_counts = impute_nan(xyzs, inplace=True)
        assert not np.any(np.isnan(xyzs)), "Imputation failed"
        xyz = xyzs.reshape(num_days * DAY, -1)  # linearize back
        imputed_locs = [i*DAY + j for i,j in imputed_locs]  # linearize indexes
        imputed_times = [i + start_time for i in imputed_locs]
        # imputed_times = [datetime.utcfromtimestamp(centis2secs(i + start_time) for i in imputed_locs]
        imputed_times_durations = list(zip(imputed_times, imputed_counts))
    else:
        imputed_times_durations = []

    if resample_hz is not None:
        scale = resample_hz / DEVICE_SAMPLE_HZ
        if resample_method == 'fft':
            xyz = resample_xyz(xyz, num=int(len(xyz)*scale))
        elif resample_method == 'linear':
            xyz = interp1d_xyz(xyz, q=int(1.0/scale))
        elif resample_method == 'decimate':
            xyz = decimate_xyz(xyz, q=int(1.0/scale))
        else:
            raise ValueError("Unrecognized resample_method {}".format(resample_method))
        xyz = xyz.astype(XYZ_DTYPE)  # the scipy packages sometimes cast things to double
    else:
        resample_method = None

    # other relevant info
    dtobj = datetime.utcfromtimestamp(centis2secs(start_time))
    month = dtobj.month - 1
    weekday = dtobj.weekday()

    #? report local time or "localized unix time"?
    info = {
        'dtype': xyz.dtype,
        'shape': xyz.shape,
        'size': xyz.size,
        'num_days': num_days,
        'start': start_time,
        'end': end_time,
        'offset': offset,
        'nonwear': nonwear_times_durations,
        'imputed': imputed_times_durations,
        'resample_hz': resample_hz,
        'resample_method': resample_method,
        'device_hz': DEVICE_SAMPLE_HZ,
        'month': month,
        'weekday': weekday,
    }

    print("\n----- info -----")
    for k,v in info.items():
        print(f"{k}: {v}")
    print("----------------\n")

    # with open(outfile, 'wb') as f:
    #     print(f"saving to {outfile}")
    #     pickle.dump({'info':info, 'xyz':xyz}, f)

    # Group into intervals of 30 secs
    xyz = xyz.reshape(-1,3000,3)

    # extractor = Extractor(fft3d=False)
    # X_feats = []
    # for i in tqdm(range(len(xyz))):
    # # for i in range(len(xyz)):
    #     base_feats = extractor.extract(xyz[i])
    #     base_feats = base_feats[[38, 55, 9, 84, 66]]
    #     quantile_feats = quantile_features(xyz[i])
    #     catch22_feats = catch22_features(xyz[i])
    #     feats = np.concatenate((base_feats, quantile_feats, catch22_feats))
    #     X_feats.append(feats)
    # X_feats = np.stack(X_feats)

    #!hack
    # xyz = xyz[:5000]

    X_feats_catch22 = []
    for i in tqdm(range(len(xyz))):
        if np.isfinite(xyz[i]).all():
            feats = catch22_features(xyz[i])
        else:
            feats = [np.nan]*7
        X_feats_catch22.append(feats)
    X_feats_catch22 = np.stack(X_feats_catch22).astype(XYZ_DTYPE)

    extractor = Extractor(fft3d=False)
    X_feats_base = []
    for i in tqdm(range(len(xyz))):
        if np.isfinite(xyz[i]).all():
            base_feats = extractor.extract(xyz[i])
            base_feats = base_feats[[38, 55, 9, 84, 66]]
            quantile_feats = quantile_features(xyz[i])
            feats = np.concatenate((base_feats, quantile_feats))
        else:
            feats = [np.nan]*8
        X_feats_base.append(feats)
    X_feats_base = np.stack(X_feats_base).astype(XYZ_DTYPE)
    Extractor.shutdown()
    
    X_feats = np.concatenate((X_feats_base, X_feats_catch22), axis=1)

    # Align days to start on Monday
    day_length = X_feats.shape[0] // num_days
    X_feats = np.concatenate((X_feats[(weekday*day_length):], X_feats[:(weekday*day_length)]))

    # # Finally, transpose to make it feature-first
    # xyz_feats = np.transpose(xyz_feats, axes=(1,0))

    # # Minimal checks
    # assert np.isfinite(xyz_feats).all(), 'NaN or Inf found during feature extraction'

    np.save(outfile, X_feats)



if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('npypath')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--impute', type=str2bool, default=False)
    parser.add_argument('--detect_nonwear', type=str2bool, default=False)
    parser.add_argument('--start_of_day', type=int, default=10*HOUR)
    parser.add_argument('--num_days', type=int, default=7)
    parser.add_argument('--resample_method', default='fft')
    parser.add_argument('--resample_hz', type=int, default=None)
    parser.add_argument('--compress', type=str2bool, default=True)
    parser.add_argument('--delete_old', type=str2bool, default=False)
    parser.add_argument('--outfile', default='tmp.pkl')
    args = parser.parse_args()

    process(
        args.npypath, args.impute, args.detect_nonwear,
        args.start_of_day, args.num_days, args.resample_method, args.resample_hz,
        args.compress, args.delete_old, args.outfile, args.seed
    )
