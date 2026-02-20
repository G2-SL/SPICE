"""
Automated flagging routines for radio interferometry measurement sets using CASA tools.
Includes temporal, channel, phase, and baseline-specific flagging algorithms.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

from casatasks import flagdata
from casatools import msmetadata, ms

# Initialize CASA tools
msmd = msmetadata()
ms_tool = ms()

def fieldinfo(scan: str, msname: str) -> float:
    """Returns the total time of a scan in minutes."""
    msmd.open(msname)
    a1 = msmd.timesforscans(scans=int(scan))
    totaltime = round((a1[-1] - a1[0]) / 60, 2)
    msmd.close()
    return totaltime

def phaseflagging(vis: str, scan: str) -> int:
    """Flags data based on phase and amplitude statistical thresholds."""
    ms_tool.open(vis, nomodify=False)
    mean, std = [], []
    baselr = baseline(vis)
    
    for i in baselr:
        a = ms_tool.statistics(column='corrected', complex_value='phase', scan=str(scan), baseline=str(i), doquantiles=False)
        mean.append(a['']['mean'])
        std.append(a['']['stddev'])
        
    a1 = ms_tool.statistics(column='corrected', complex_value='amplitude', scan=str(scan), doquantiles=False)['']
    mean1, std1 = a1['mean'], a1['stddev']
    
    ms_tool.select({'scan_number': [int(scan)]})
    d = ms_tool.getdata(["corrected_phase", "corrected_amplitude", "flag"], ifraxis=True)
    phase, amp = d['corrected_phase'], d['corrected_amplitude']
    
    for i in range(len(baselr)):
        d['flag'][:, :, i, :][np.where((phase[:, :, i, :] < mean[i] - 5 * std[i]) | (phase[:, :, i, :] > mean[i] + 5 * std[i]))] = True
        
    d['flag'][np.where((amp < mean1 - 5 * std1) | (amp > mean1 + 5 * std1))] = True
    d['flag'][np.where((amp < 0.1) | (amp > 2.5))] = True
    
    ms_tool.putdata(d)
    ms_tool.close()
    return True

def baseline(msfile: str, badant: list = None, badbase: list = None) -> list:
    """Generates a list of valid baselines, excluding specified bad antennas/baselines."""
    if badant is None: badant = []
    if badbase is None: badbase = []
    
    msmd.open(msfile)
    antenna_names = msmd.antennanames()
    a = msmd.baselines()
    msmd.close()
    
    baselr = []
    for i in range(len(antenna_names)):
        for j in range(len(antenna_names)):
            if j > i:
                baselr.append(f'{antenna_names[i]}&{antenna_names[j]}')
                
    nbase = np.array(list(a[np.triu_indices(len(antenna_names), k=1)]))
    baselr = np.array(baselr)[np.where(nbase == True)]
    
    for i in baselr:
        for j in badant:
            if j in i:
                badbase.append(i)
                
    return [item for item in baselr if item not in badbase]

def flagantbase(msfile: str, cscan: list, channel: int, antenna: list, sigma: float = 2.2) -> list:
    """Flags specific antennas and baselines based on amplitude statistics."""
    ants, scanstd = [], []
    ms_tool.open(msfile)
    
    for j in cscan:
        bbline, antmean, antstd = [], [], []
        spw = f'0:{int(0.4 * channel)}~{int(0.6 * channel)}'
        
        for i in antenna:
            a = ms_tool.statistics(column='data', complex_value='amplitude', spw=spw, scan=str(j), baseline=i, doquantiles=False)['']
            if a['npts'] == 0:
                bbline.append(i)
                antmean.append(np.nan)
                antstd.append(np.nan)
            else:
                antmean.append(a['mean'])
                antstd.append(a['stddev'])
                
        antmean = np.array(antmean)
        antstd = np.array(antstd)
        antenna_arr = np.array(antenna)
        
        a1, a2 = np.nanmedian(antmean), np.nanstd(antmean)
        ant1 = antenna_arr[np.where((antmean < a1 - sigma * a2) | (antmean > a1 + sigma * a2))]
        
        a1, a2 = np.nanmedian(antstd), np.nanstd(antstd)
        ant2 = antenna_arr[np.where(antstd > a1 + sigma * a2)]
        
        ants.append(bbline + list(set(list(ant1) + list(ant2))))
        scanstd.append(np.std(antmean))
        
    ms_tool.close()
    if len(ants) == 0:
        return []
    return list(set.intersection(*map(set, ants)))

def flagant(msfile: str, cscan: list, channel: int, antenna: list) -> list:
    """Identifies and logs consistently bad antennas across scans."""
    ants = []
    ms_tool.open(msfile)
    for j in cscan:
        bbline = []
        spw = f'0:{int(0.3 * channel)}~{int(0.7 * channel)}'
        for i in antenna:
            a = ms_tool.statistics(column='data', complex_value='amplitude', spw=spw, scan=str(j), baseline=i, correlation='rr', doquantiles=False)['']
            if a['npts'] == 0:
                bbline.append(i)
        ants.append(bbline)
    ms_tool.close()
    
    badant = list(set.intersection(*map(set, ants)))
    badantscan, badscan = [], []
    for i in range(len(cscan)):
        a = list(set(ants[i]) - set(badant))
        if len(a) > 0:
            badantscan.append(a)
            badscan.append(cscan[i])
            
    with open('flagging_log.txt', "a") as f:
        f.write(fr'# flagged antenna/baseline {[badant, badantscan, badscan]}' + '\n')
    return [badant, badantscan, badscan]

def flagtime(msfile: str, scan: str) -> list:
    """Flags data in the time domain based on amplitude deviations."""
    msmd.open(msfile)
    times = msmd.timesforscan(scan)
    intime = round(msmd.exposuretime(scan=scan)['value'])
    msmd.done()
    
    antenna = np.array(range(len(times)))
    ms_tool.open(msfile)
    antmean, antstd = [], []
    for i in antenna:
        x1 = Time(times[i] / 86400, format='mjd').iso
        a = ms_tool.statistics(column='data', time=x1.replace("-", "/").replace(' ', '/'), complex_value='amplitude', doquantiles=False)['']
        antmean.append(a['mean'])
        antstd.append(a['stddev'])
    ms_tool.close()
    
    antmean, antstd = np.array(antmean), np.array(antstd)
    sigma = 3
    
    a1, a2 = np.nanmedian(antmean), np.nanstd(antmean)
    index1 = np.where((antmean < a1 - sigma * a2) | (antmean > a1 + sigma * a2))[0]
    antmean[index1] = np.nan
    
    a1, a2 = np.nanmedian(antmean), np.nanstd(antmean)
    index2 = np.where((antmean < a1 - sigma * a2) | (antmean > a1 + sigma * a2))[0]
    index = list(index1) + list(index2)
    
    antstd[index] = np.nan
    a1, a2 = np.median(antstd), np.std(antstd)
    index1 = np.where((antstd < a1 - sigma * a2) | (antstd > a1 + sigma * a2))[0]
    badbase = list(set(index + list(index1)))
    
    for i in badbase:
        x1 = Time(times[i] / 86400, format='mjd').iso
        flagdata(msfile, mode='manual', timerange=x1.replace("-", "/").replace(' ', '/'))
        times[i] = 0

    if len(times) - len(badbase) > 1:
        slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(times))
        times = times[slices]
        begt = Time((times[0] - intime + 1) / 86400, format='mjd').iso
        endt = Time((times[-1] + intime - 1) / 86400, format='mjd').iso
        timerange = begt.replace("-", "/").replace(' ', '/') + '~' + endt.replace("-", "/").replace(' ', '/')
        return [scan, timerange]
    else:
        return [scan]

def flagchannel(msfile: str, channel: int, scan: str) -> np.ndarray:
    """Flags specific frequency channels based on amplitude standard deviations."""
    antmean, antstd, index4 = [], [], []
    ms_tool.open(msfile)
    ms_tool.select({'scan_number': [int(scan)]})
    for i in range(channel):
        a = ms_tool.statistics(column='data', spw=f'0:{i}', complex_value='amplitude', doquantiles=False)['']
        if a['mean'] == 0:
            antmean.append(np.nan)
            antstd.append(np.nan)
            index4.append(i)
        else:
            antmean.append(a['mean'])
            antstd.append(a['stddev'])
    ms_tool.close()
    
    antmean, antstd = np.array(antmean), np.array(antstd)
    sigma = 3
    a1, a2 = np.nanmedian(antmean), np.nanstd(antmean)
    index1 = np.where((antmean < a1 - sigma * a2) | (antmean > a1 + sigma * a2))[0]
    antmean[index1] = np.nan
    
    a1, a2 = np.nanmedian(antmean), np.nanstd(antmean)
    index2 = np.where((antmean < a1 - sigma * a2) | (antmean > a1 + sigma * a2))[0]
    index = list(index1) + list(index2)
    
    antstd[index] = np.nan
    a1, a2 = np.median(antstd), np.std(antstd)
    index1 = np.where((antstd < a1 - sigma * a2) | (antstd > a1 + sigma * a2))[0]
    badbase = list(set(index + list(index1)))
    
    chan = np.linspace(0, channel - 1, channel)
    for i in badbase:
        flagdata(msfile, mode='manual', scan=str(scan), spw=f'0:{i}')
        chan[i] = np.nan
    return chan

def baselflag(filen: str, scan: str, sigma: float, column: str, flag1: int) -> int:
    """Performs iterative flagging on individual baselines and generates diagnostic plots."""
    baselr = baseline(filen)
    ba = np.array(baselr)
    bs5, bs6 = [], []
    ms_tool.open(filen)
    
    for j in baselr:
        a = ms_tool.statistics(column=column, complex_value='amplitude', scan=str(scan), baseline=str(j), doquantiles=False)
        bs5.append(a['']['mean'])
        bs6.append(a['']['stddev'])
    ms_tool.close()
    
    a23 = [bs5, bs6]
    for ks in range(flag1):
        bs = np.array(a23[ks])
        baseline1 = ba[np.where(bs > 0)]
        bs3 = bs[np.where(bs > 0)]
        
        if len(bs3) == 0: continue
            
        a1, a2 = np.mean(bs3), np.std(bs3)
        a12 = baseline1[np.where((bs3 < a1 - sigma * a2) | (bs3 > a1 + sigma * a2))]
        baseline2 = baseline1[np.where((bs3 > a1 - sigma * a2) & (bs3 < a1 + sigma * a2))]
        bs1 = bs3[np.where((bs3 > a1 - sigma * a2) & (bs3 < a1 + sigma * a2))]
        
        a1, a2 = np.mean(bs1), np.std(bs1)
        a13 = baseline2[np.where((bs1 < a1 - sigma * a2) | (bs1 > a1 + sigma * a2))]
        a14 = a12.tolist() + a13.tolist()
        ba1 = ';'.join(map(str, a14))
        
        if len(a14) > 0:
            flagdata(vis=filen, mode='manual', antenna=ba1, scan=str(scan))
            plt.plot(bs3)
            bs2 = bs1[np.where((bs1 > a1 - sigma * a2) & (bs1 < a1 + sigma * a2))]
            plt.plot(bs2)
            plt.savefig(f'baseflag_scan{scan}.jpg')
            plt.close()
    return True