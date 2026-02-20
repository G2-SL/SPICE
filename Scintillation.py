"""
scintillation.py

Contains functions for generating dynamic spectra from measurement sets
and fitting 2D Gaussians for pulsar scintillation analysis.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import models, fitting

from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from casatasks import listobs
from casatools import ms

# Initialize CASA tools locally for this module
ms_tool = ms()

def dynamicsp(file: str) -> list:
    """Extracts the dynamic spectra from a measurement set."""
    ms_tool.open(file)
    ms_tool.iterinit(interval=0.0, maxrows=1)
    ms_tool.iterorigin()
    ds1, ds2 = [], []
    
    for _ in range(10000):
        d = ms_tool.getdata(["corrected_real", "flag"], ifraxis=True)
        k = np.where(d['flag'] == False, 1, np.nan)
        visi = d['corrected_real'] * k
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av = np.nanmean(visi[:, :, :], axis=0)
            nchannel, nbaseline = av.shape
            a1 = np.arange(int(nbaseline / 2)) * 2
            a2 = list(set(np.arange(nbaseline)) - set(a1))
            fe1 = np.nanmean(av[:, a1], axis=1)
            fe2 = np.nanmean(av[:, a2], axis=1)
            
        ds1.append(fe1)
        ds2.append(fe2)
        x = ms_tool.iternext()
        if not x:
            break
            
    ms_tool.close()
    return [np.array(ds1), np.array(ds2)]

def polyfit(data: np.ndarray) -> np.ndarray:
    """Applies a 2D polynomial fit to remove broad bandpass/time structures."""
    nan_mask = np.isnan(data)
    data1 = np.nan_to_num(data, nan=np.nanmedian(data))
    fit_p = fitting.LevMarLSQFitter()
    p_init = models.Polynomial2D(degree=4)
    M, N = data.shape
    y, x = np.mgrid[:M, :N]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
        p = fit_p(p_init, x, y, data1)
        
    result = data - p(x, y)
    result[nan_mask] = 0
    return result

def fittin(field: str, scan: str, m0: float, m1: float, isnr: float, flux: float, msfile: str, 
           ii: int, chanwidth: float, intime: float, ctrlfreq: float, bandwidth: float, dire: str) -> int:
    """
    Performs a 2D autocorrelation and fits a 2D Gaussian to extract scintillation timescales 
    and bandwidths. Saves diagnostic plots and appends results to a log file.
    """
    import os # imported locally to prevent namespace clutter
    
    sourceid = f'{scan}.{ii}'
    c = SkyCoord(ra=m0 * u.deg, dec=m1 * u.deg, frame='icrs')
    a = c.to_string('hmsdms').split(' ')
    RA, DEC = a[0], a[1]
    
    data = dynamicsp(msfile)
    data1 = polyfit(data[0])
    data2 = polyfit(data[1])
    dsnoise = round(np.std((data1 + data2) / 2) * 1e3, 2)

    y_cor = signal.correlate(data1, data2, 'same')
    a_arr = np.ones(data1.shape)
    acor = signal.correlate(a_arr, a_arr, 'same')
    y_cor = y_cor / acor
    y_cor15 = median_filter(y_cor, size=3)
    M, N = y_cor15.shape
    y_cor14 = y_cor[int(M/2)-7:int(M/2)+7, int(N/2)-7:int(N/2)+7]
    y_cor15[int(M/2)-7:int(M/2)+7, int(N/2)-7:int(N/2)+7] = 0
    ppr = round(max(y_cor15.max(), abs(y_cor15.min())) / y_cor14.max(), 2)
        
    a_flat = y_cor.flatten()
    a1_flat = a_flat[np.where((a_flat > np.mean(a_flat) - 3 * np.std(a_flat)) & (a_flat < np.mean(a_flat) + 3 * np.std(a_flat)))]
    no = np.std(a1_flat)

    y, x = np.mgrid[:M, :N]
    y_cor1 = y_cor.ravel()
    ele = np.where(y_cor1 == y_cor1.max())[0][0]
    y_cor2 = np.delete(y_cor1, ele)
    x1, x2 = x.ravel(), np.delete(x.ravel(), ele)
    y1, y2 = y.ravel(), np.delete(y.ravel(), ele)
    
    def twoD_Gaussian(xdata_tuple, amplitude, sigma_x, sigma_y, theta):
        x, y = xdata_tuple
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = (np.sin(2 * theta)) / (4 * sigma_x**2) - (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        g = amplitude * np.exp(- (a * ((x - N/2)**2) + 2 * b * (x - N/2) * (y - M/2) + c * ((y - M/2)**2)))
        return g
        
    with open('scintillation_log.csv', 'ab') as f:
        np.savetxt(f, [sourceid, RA, DEC, isnr, flux, ppr, dsnoise], fmt='%18s', delimiter=',', newline='')
        f.write(b'\n')
        
    if (ppr > 0) & (ppr <= 1):
        np.save(f'{sourceid}_data1.npy', data[0])
        np.save(f'{sourceid}_data2.npy', data[1])
        
        max_val = np.percentile(y_cor2, 95)
        y_cor2 = y_cor2 / max_val
        initial_guess = (y_cor2.max() / 2, 1.2, 1.2, 1e-4)
        
        try:
            popt, pcov = curve_fit(twoD_Gaussian, (x2, y2), y_cor2, p0=initial_guess,
                                   bounds=([0, 0, 0, -np.pi/4], [y_cor2.max() * 2, 1.5 * N/4, 1.5 * M/4, np.pi/4]))
            pc3 = np.sqrt(np.diag(pcov))
            figure = True
        except RuntimeError:
            figure = False
            with open('scintillation_log.csv', 'ab') as f:
                np.savetxt(f, ['#nan'], fmt='%18s', delimiter=',', newline='')
                f.write(b'\n')
        
        if figure:
            th, eth = popt[3], pc3[3]
            a, b = popt[1], popt[2]
            da, db = pc3[1], pc3[2]
            
            scp = [a * b * (b**2 * np.cos(th)**2 + a**2 * np.sin(th)**2)**(-0.5),
                   a * b * (a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2)**(-0.5)]
                   
            # Error propagation (simplified variables for readability)
            term1 = (-a**2 * b * np.sin(th)**2 / (a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2)**(3/2) + b / np.sqrt(a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2))**2 * da**2
            term2 = (-a * b**2 * np.cos(th)**2 / (a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2)**(3/2) + a / np.sqrt(a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2))**2 * db**2
            term3 = (a * b * (-a**2 * np.sin(th) * np.cos(th) + b**2 * np.sin(th) * np.cos(th)) / (a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2)**(3/2))**2 * eth**2
            esf = np.sqrt(term1 + term2 + term3)
            
            term4 = (-a**2 * b * np.cos(th)**2 / (a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2)**(3/2) + b / np.sqrt(a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2))**2 * da**2
            term5 = (-a * b**2 * np.sin(th)**2 / (a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2)**(3/2) + a / np.sqrt(a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2))**2 * db**2
            term6 = (a * b * (a**2 * np.sin(th) * np.cos(th) - b**2 * np.sin(th) * np.cos(th)) / (a**2 * np.cos(th)**2 + b**2 * np.sin(th)**2)**(3/2))**2 * eth**2
            est = np.sqrt(term4 + term5 + term6)

            sfreq = round(np.sqrt(2 * np.log(2)) * scp[0] * chanwidth, 3)
            esfreq = round(np.sqrt(2 * np.log(2)) * esf * chanwidth, 3)
            stime = round(np.sqrt(2) * scp[1] * intime, 3)
            estime = round(np.sqrt(2) * est * intime, 3)

            MJD = round(listobs('out3.ms')['BeginTime'], 4)
            
            def Gauss(x, a, x0, sigma):
                return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                
            ty = y_cor[:, int(N/2)]
            fy = y_cor[int(M/2)]
            
            # Sub-fitting for Time parameter
            ele_t = np.where(ty == np.max(ty))[0][0]
            x1_t = range(len(ty))
            ty = np.delete(ty, ele_t)
            x2_t = np.delete(x1_t, ele_t)
            max_val1 = np.max(ty) if np.max(ty) > 0 else 1
            ty, medty = ty / max_val1, np.median(ty / max_val1)
            ty = ty - medty
            
            z1 = 2
            for i in range(int(M/2) - 10):
                a_val = ty[int(M/2) + i + 1]
                z1 = i + 2
                if a_val < 0.5 - medty / 2 or i >= int(M/2) - 11:
                    break
                    
            poptty, pcovty = curve_fit(Gauss, x2_t, ty, p0=[1 - medty, M/2, 1.2], bounds=([0, M/2 - 0.4, 0], [1 - medty, M/2 + 0.4, z1]))
            
            # Sub-fitting for Freq parameter
            ele_f = np.where(fy == np.max(fy))[0][0]
            x1_f = range(len(fy))
            fy = np.delete(fy, ele_f)
            x2_f = np.delete(x1_f, ele_f)
            max_val2 = np.max(fy) if np.max(fy) > 0 else 1
            fy, medfy = fy / max_val2, np.median(fy / max_val2)
            fy = fy - medfy
            
            z1 = 2
            for i in range(int(N/2) - 10):
                a_val = fy[int(N/2) + i + 1]
                z1 = i + 2
                if a_val < 0.5 - medfy / 2 or i >= int(N/2) - 11:
                    break
                    
            poptfy, pcovfy = curve_fit(Gauss, x2_f, fy, p0=[1 - medfy, N/2, 1.2], bounds=([0, N/2 - 0.4, 0], [1 - medfy, N/2 + 0.4, z1]))
            
            pcty, pcfy = np.sqrt(np.diag(pcovty)), np.sqrt(np.diag(pcovfy))
            sfreq1 = round(np.sqrt(2 * np.log(2)) * poptfy[2] * chanwidth, 3)
            esfreq1 = round(np.sqrt(2 * np.log(2)) * pcfy[2] * chanwidth, 3)
            stime1 = round(np.sqrt(2) * poptty[2] * intime, 3)
            estime1 = round(np.sqrt(2) * pcty[2] * intime, 3)
            
            ampt, eampt = round((poptty[0] + medty) * max_val1 * 1e6, 3), round(pcty[0] * max_val1 * 1e6, 3)
            ampf, eampf = round((poptfy[0] + medfy) * max_val2 * 1e6, 3), round(pcfy[0] * max_val2 * 1e6, 3)
            modul = round(np.sqrt(max(ampt, ampf)) / flux, 2)
            snr = round(max((poptfy[0] + medfy) * max_val2, (poptty[0] + medty) * max_val1) * np.sqrt(np.pi * poptty[2] * poptfy[2] / 2) / no, 1)
            
            from flagging import fieldinfo1 # local import assuming files are in same directory
            fib = [dsnoise, ppr, sfreq1, esfreq1, ampf, eampf, stime1, estime1, ampt, eampt, modul, snr, 
                   round(ctrlfreq * 1e-6), bandwidth, chanwidth, fieldinfo1(scan, 'out4.ms'), intime, MJD, M * N]
                   
            # Plotting logic
            fdata = twoD_Gaussian((x, y), *popt).reshape(M, N) * max_val
            fig = plt.figure(figsize=(20, 20))
            
            plt.subplot(4, 2, 1)
            plt.imshow(data1, aspect='auto', vmin=data1.min()/5, vmax=data1.max()/2, cmap='seismic', origin='lower')
            plt.xlabel('frequency'); plt.ylabel('time'); plt.colorbar()
            
            plt.subplot(4, 2, 2)
            plt.imshow(data2, aspect='auto', vmin=data2.min()/5, vmax=data2.max()/2, cmap='seismic', origin='lower')
            plt.xlabel('frequency'); plt.ylabel('time'); plt.colorbar()

            ax = plt.subplot(4, 2, 3)
            plt.imshow(y_cor, cmap='inferno', aspect='auto', origin='lower')
            plt.colorbar()
            ellipse = mpl.patches.Ellipse((N/2, M/2), width=popt[1]*5, height=popt[2]*3, angle=(popt[3]*180/np.pi), edgecolor='w', facecolor='none')
            ax.add_patch(ellipse)
            plt.xlabel('frequency'); plt.ylabel('time')

            plt.subplot(4, 2, 4)
            plt.xlabel('Time lag', fontsize=20)
            ty_plot, fty = y_cor[:, int(N/2)], fdata[:, int(N/2)]
            x1_plot = np.arange(-len(ty_plot+1)/2, len(ty_plot-1)/2)
            plt.scatter(x1_plot, ty_plot, s=2, c='k', label='data')
            plt.plot(x1_plot, fty, 'r-', label='fit'); plt.legend()

            plt.subplot(4, 2, 5)
            plt.xlabel('Frequency lag', fontsize=20)
            fy_plot, ffy = y_cor[int(M/2), :], fdata[int(M/2), :]
            x1_plot = np.arange(-len(fy_plot+1)/2, len(fy_plot-1)/2)
            plt.scatter(x1_plot, fy_plot, s=2, c='k', label='data')
            plt.plot(x1_plot, ffy, 'r-', label='fit'); plt.legend()

            plt.subplot(4, 2, 6)
            plt.xlabel('Time lag', fontsize=20)
            x1_plot = np.arange(-len(ty_plot+1)/2, len(ty_plot-1)/2)
            plt.plot(x1_plot, ty_plot, 'b:', label='data')
            plt.plot(x1_plot, (Gauss(range(len(ty_plot)), *poptty) + medty) * max_val1, 'r-', label='fit'); plt.legend()

            plt.subplot(4, 2, 7)
            plt.xlabel('Frequency lag', fontsize=20)
            x1_plot = np.arange(-len(fy_plot+1)/2, len(fy_plot-1)/2)
            plt.plot(x1_plot, fy_plot, 'b:', label='data')
            plt.plot(x1_plot, (Gauss(range(len(fy_plot)), *poptfy) + medfy) * max_val2, 'r-', label='fit'); plt.legend()

            fig.savefig(f'source_{ii}.jpg')
            plt.close(fig)
            
            if ((sfreq > 5 * esfreq) & (stime > 3 * estime)) | ((sfreq > 3 * esfreq) & (stime > 5 * estime)):
                os.chdir(dire)
                with open('scintillation_log.csv', 'ab') as f:
                    np.savetxt(f, [sourceid, m0, m1, isnr, flux] + fib, fmt='%18s', delimiter=',', newline='')
                    f.write(b'\n')
                os.chdir(os.path.join(dire, scan))
    return True