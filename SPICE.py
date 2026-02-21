#!/usr/bin/env python3
"""
Handles MS ingestion, metadata extraction, primary calibration, imaging and source extraction for target fields.
"""

import os
import glob
import time
import shutil
import argparse
import numpy as np
from astropy.table import Table
from astropy import units as u

# External modules
import bdsf
import casatasks as ct
from casatasks import listobs, flagdata, gaincal, bandpass, applycal
from casatools import msmetadata, ms, image, regionmanager

# Assume flagging and scintillation modules are in the same directory
from flagging import flagtime, flagchannel, flagantbase, flagant, baselflag, phaseflagging, fieldinfo
from scintillation import fittin

# Initialize CASA tools
msmd = msmetadata()
ms_tool = ms()
ia = image()
rg = regionmanager()

def process_target_field(field, scan, res, refant, base_dir, dire, channel, chanwidth, intime, ctrlfreq, bandwidth):
    """
    Handles self-calibration, imaging (tclean), and PyBDSF source extraction 
    for a specific target field and scan.
    Returns True on success, False on soft failure.
    """
    file2 = glob.glob("*.fits")
    msfile = os.path.join(dire, 'out.ms')
    
    if len(file2) == 0:
        # Assuming baseline and flagant are imported from flagging module
        from flagging import baseline, flagant
        baselr = baseline(msfile)
        badbase = flagant(msfile, [int(scan)], channel, baselr)[0]
        baselr = [item for item in baselr if item not in badbase]
        ba = ';'.join(map(str, baselr))
        
        ct.split(vis=msfile, outputvis="out1.ms", datacolumn="corrected", scan=str(scan), antenna=ba)
        ttime = fieldinfo(scan, 'out1.ms')
        
        os.chdir(dire)
        with open('pipeline_log.txt', "a") as f:
            f.write(f'\n# {field} having a scan with duration {ttime} mins')
            
        target_scan_dir = os.path.join(dire, str(scan))
        os.chdir(target_scan_dir)
        
        # Initial Clean
        ct.tclean(vis='out1.ms', imagename='field_image', spw='', specmode='mfs', deconvolver='hogbom',
                  gridder='standard', imsize=[2880], cell=[f'{res}arcsec'], weighting='briggs', robust=0.5, 
                  threshold='0mJy', niter=5000, interactive=False, savemodel='modelcolumn')
                  
        ia.open("field_image.residual")
        r1 = rg.box([1040, 1040], [1340, 1340])
        a = ia.statistics(region=r1)
        ia.close()
        threshold = f"{round(a['rms'][0]*3e3, 3)}mJy"
        
        # First round self-phase cal
        gaincal(vis='out1.ms', caltable="phase.cals", solint='int', calmode="p", refant=refant, gaintype="T")
        if not os.path.isdir('phase.cals'):
            os.chdir(dire)
            with open('pipeline_log.txt', "a") as f:
                f.write(' and phase1.cals not found\n')
            shutil.rmtree(scan)
            return False
            
        applycal(vis='out1.ms', gaintable=["phase.cals"], interp="linear")
        ct.split(vis="out1.ms", outputvis="out2.ms", datacolumn="corrected")
        
        ct.tclean(vis='out2.ms', imagename='field_image1', spw='', specmode='mfs', deconvolver='hogbom',
                  gridder='standard', imsize=[2880], cell=[f'{res}arcsec'], weighting='briggs', robust=0.5, 
                  threshold=threshold, niter=5000, interactive=False, savemodel='modelcolumn')
                  
        # Second round self-phase cal
        gaincal(vis="out2.ms", caltable="phase1.cals", solint='int', calmode="p", refant=refant, gaintype="T")
        if not os.path.isdir('phase1.cals'):
            os.chdir(dire)
            with open('pipeline_log.txt', "a") as f:
                f.write(' and phase2.cals not found\n')
            shutil.rmtree(scan)
            return False
            
        applycal(vis="out2.ms", gaintable=["phase1.cals"], interp="linear")
        
        # Create final image
        ct.split(vis="out2.ms", outputvis="out3.ms", datacolumn="corrected")
        ct.tclean(vis='out3.ms', imagename='field_image2', spw='', specmode='mfs', deconvolver='hogbom',
                  gridder='standard', imsize=[2880], cell=[f'{res}arcsec'], weighting='briggs', robust=0.5, 
                  threshold=threshold, niter=5000, interactive=False, savemodel='modelcolumn')
                  
        ct.uvsub(vis='out3.ms')
        ct.impbcor(imagename='field_image2.image', pbimage='field_image2.pb', outfile='field_image2.pbcor.image')
        
        # Export to FITS and run PyBDSF
        ia.open("field_image2.pbcor.image")
        fits_file = f"{field}{scan}.fits"
        ia.tofits(fits_file, overwrite=True)
        ia.close()
        
        img = bdsf.process_image(fits_file, adaptive_rms_box=True, output_opts=True, bbs_patches='source',
                                 rms_box_bright=(18, 6), advanced_opts=True, spline_rank=1,
                                 flagging_opts=True, flag_maxsize_bm=1.3, flag_smallsrc=True,
                                 thresh='hard', adaptive_thresh=30, rms_box=(240, 70), quiet=True)
        
        img.write_catalog(outfile='sourc.fits', format='fits', clobber=True, catalog_type='srl')
        
        if os.path.isfile('sourc.fits'):
            a_tbl = Table.read('sourc.fits', format='fits')
            a_tbl.add_column(a_tbl['Total_flux'] / a_tbl['Isl_rms'], name='SNR')
            a_tbl.sort(['SNR'], reverse=True)
            x = np.where(a_tbl['SNR'] >= 4)[0]
            a_tbl[x].write('source.fits', format='fits', overwrite=True)
            os.remove('sourc.fits')
            
        if os.path.isfile(f'{fits_file}.pybdsf.log'):
            os.remove(f'{fits_file}.pybdsf.log')

        # RMS Calculation
        ia.open("field_image2.residual")
        r1 = rg.box([1040, 1040], [1340, 1340])
        a_stat = ia.statistics(region=r1)
        ia.close()
        irms = round(a_stat['rms'][0] * 1e6, 1)
        
        os.chdir(dire)
        with open('pipeline_log.txt', "a") as f:
            f.write(f' and RMS {irms} uJy\n')
            
        os.chdir(target_scan_dir)
        if irms > 1e6:
            os.chdir(dire)
            shutil.rmtree(scan)
            return False
            
        # Call 2D Gaussian fitting for scintillation analysis
        if not os.path.isfile('source.fits'):
            os.chdir(dire)
            with open('pipeline_log.txt', "a") as f:
                f.write('# no source present\n')
            return False
            
        atm = Table.read('source.fits', format='fits')
        jstart = len(np.loadtxt('pipeline_log.txt', dtype=str)) if os.path.isfile('pipeline_log.txt') else 0
        
        for i in range(jstart, len(atm['RA'])):
            m0, m1 = round(atm['RA'][i], 7), round(atm['DEC'][i], 7)
            isnr = round(atm['SNR'][i], 1)
            flux = round(atm['Total_flux'][i] * 1e3, 2)
            
            if not os.path.isdir('out4.ms'):
                ct.phaseshift(vis='out3.ms', outputvis='out4.ms', phasecenter=f'J2000 {m0}deg {m1}deg')
            
            # Execute imported fittin function
            fittin(field, scan, m0, m1, isnr, flux, 'out4.ms', i, chanwidth, intime, ctrlfreq, bandwidth, dire)
            shutil.rmtree('out4.ms')
            
    return True

def data_analysis( target_dir, catrms=True, flux_cal='calc', refant='calc', badscan=None, badant=None, send=True):
    """
    SPICE execution function.
    Returns True on full success, raises exceptions on critical failures, or returns False on soft missing data.
    """
    if badscan is None: badscan = []
    if badant is None: badant = []
    base_dir = os.path.dirname(target_dir)
    folder = os.path.basename(target_dir)
    
    start = time.time()
    
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Error: Target directory '{target_dir}' does not exist.")
        
    os.chdir(target_dir)
    
    # 1. File ingestion and FITS conversion
    a12 = glob.glob("ou*.ms")
    msfile = 'ou.ms'
    if len(a12) == 0:
        files = glob.glob("*.fits")
        if len(files) == 0:
            files1 = glob.glob("*.lta")
            if files1:
                # Assuming listscan and gvfits are standard external scripts mapped in PATH
                os.system(f'listscan {files1[0]}')
                os.system(f'gvfits {files1[0][:-4]}.log')
                ct.importgmrt(fitsfile='TEST.FITS', vis=msfile)
                os.rename('TEST.FITS', f'{files1[0][:-4]}.fits')
                os.remove(files1[0])
            else:
                raise FileNotFoundError("Critical Error: No .lta, .fits, or .ms files found in the target directory.")
        else:
            ct.importgmrt(fitsfile=files[0], vis=msfile)
        listobs(vis=msfile, listfile='obslist1.txt', verbose=True, overwrite=True)
    else: 
        msfile = a12[0]

    # 2. Metadata Extraction
    msmd.open(msfile)
    channel = msmd.nchan(0)
    field = msmd.fieldnames()
    bandwidth = round(msmd.bandwidths(0) * 1e-6) or round(msmd.bandwidths(0))
    chanwidth = round(bandwidth * 1e3 / channel, 1)
    ctrlfreq = msmd.meanfreq(0)
    
    cals_list_path = os.path.join(base_dir, 'cals.list')
    vlacals = np.loadtxt(cals_list_path, dtype='str') if os.path.isfile(cals_list_path) else []
    
    cal_field, tar_field = [], []
    for f in field:
        if f in vlacals: cal_field.append(f)
        else: tar_field.append(f)

    if not cal_field or not tar_field:
        os.chdir(base_dir)
        shutil.rmtree(folder)
        error_msg = f'{folder} # missing calibrators or targets. Cals: {cal_field}, Targets: {tar_field}\n'
        with open('pulsar_summary.txt', "a") as f:
            f.write(error_msg)
        msmd.close()
        print(error_msg)
        return False

    cal_scan, tar_scan = [], []
    for i in cal_field: cal_scan += list(msmd.scansforfield(i))
    for i in tar_field: tar_scan += list(msmd.scansforfield(i))
    bandwidth=round(msmd.bandwidths(0)*1e-6)
    if bandwidth==0: bandwidth=round(msmd.bandwidths(0))
    channel=msmd.nchan(0); chanwidth=round(bandwidth*1e3/channel,1); ctrlfreq=msmd.meanfreq(0)
    intime=msmd.exposuretime(scan=cal_scan[0])['value']; freqs=msmd.chanfreqs(0)
    msmd.close()
    
    f os.path.isfile('filename')==False:
        flagdata(msfile,mode="clip", clipminmax=[0,100],clipoutside=True,clipzeros=True)
        in_ch=int(1*channel/100)
        badchans=['0:'+str(i) for i in range(in_ch)]+['0:'+str(i) for i in range(channel-in_ch,channel)]
        badchans=[]
        rfifreq=[0.36E09,0.3796E09,0.486E09,0.49355E09,0.8808E09,0.885596E09,0.7646E09,0.769092E09] # always bad
        msmd.open(msfile)
        freqs=msmd.chanfreqs(0)
        antenna_names=np.array(['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'E02', 'E03', 'E04', 'E05', 'E06', 'S01', 'S02', 'S03', 'S04', 'S06', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06'])
        
        for j in range(0,len(rfifreq)-1,2):
            for i in range(0,len(freqs)):
                if (freqs[i] > rfifreq[j] and freqs[i] < rfifreq[j+1]):
                        badchans.append('0:'+str(i))
        if badchans!=[]:
            chanflag = str(', '.join(badchans))
            flgcmd = ["mode='manual' spw='%s'" % (chanflag)]
            flagdata(msfile,mode='list', inpfile=flgcmd)

        total_scans=range(1,msmd.nscans()+1)
        for i in total_scans:
            totime=len(msmd.timesforscans(scans=i))
            if totime>10:
                flagdata(msfile, mode='quack',scan=str(i),quackinterval=2, quackmode='beg')
                flagdata(msfile, mode='quack',scan=str(i),quackinterval=2, quackmode='endb')
            if i in cal_scan:
                if totime>25:
                    flagdata(msfile, mode='quack',scan=str(i),quackinterval=intime*6, quackmode='beg')
            if i in tar_scan:
                if totime<25: badscan.append(i)
            if totime<2:
                badscan.append(i)
        msmd.close()
        
        cal_scan=list(set(cal_scan)-set(badscan))
        
        total_scans=list(set(total_scans)-set(badscan))
        chan=[]
        for i in total_scans:
            a11=flagtime(msfile,i)
            if len(a11)==2:
                cha1=flagchannel(msfile,channel,i)
                chan.append(cha1)
                ct.split(msfile,outputvis='outcal'+str(i)+'.ms',datacolumn="data",timerange=a11[1],correlation="RR,LL")
                if os.path.isdir('ou.ms.flagversions'): shutil.rmtree('ou.ms.flagversions')
        shutil.rmtree('ou.ms')
        file=glob.glob('outcal'+'*'+'.ms'); msfile='out3.ms'
        ct.concat(vis=file,concatvis=msfile)
        for i in file: shutil.rmtree(i)
        msmd.open(msfile)
        cal_scan=[];tar_scan=[]
        for i in cal_field:
            cal_scan+=list(msmd.scansforfield(i))
        for i in tar_field:
            tar_scan+=list(msmd.scansforfield(i))
        msmd.close()
        total_scans=cal_scan+tar_scan
        goodsc=','.join(map(str, total_scans))
        cen_antenna_names=['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14']
        out_antenna_names=['E02', 'E03', 'E04', 'E05', 'E06',
                 'S01', 'S02', 'S03', 'S04', 'S06',
                 'W01', 'W02', 'W03', 'W04', 'W05', 'W06']
        
        badant+=flagantbase(msfile,cal_scan,channel,antenna_names,sigma=1.8)
        badcenant=[]; badoutant=[]
        for i in badant:
            for j in cen_antenna_names:
                if j==i:
                    badcenant.append(i)
            for j in out_antenna_names:
                if j==i:
                    badoutant.append(i)
        antenna_names=[item for item in antenna_names if item not in badant]
        goodcentant=list(set(cen_antenna_names)-set(badcenant))
        goodoutant=list(set(out_antenna_names)-set(badoutant))
        baselr=[]
        for i in range(len(antenna_names)):
            for j in range(len(antenna_names)):
                if j>i:
                    if (antenna_names[i] in goodcentant) & (antenna_names[j] in goodcentant):
                        pass
                    else:
                        baselr.append(f'{antenna_names[i]}&{antenna_names[j]}')        
        ba=';'.join(map(str, baselr))
        
        chan=np.array(chan); chan=np.nanmean(chan,axis=0); chan=np.nan_to_num(chan,nan=0)
        slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(chan))
        chan=chan[slices]; channel=len(chan)
        spw='0:'+str(int(chan[0]))+'~'+str(int(chan[-1]))
        msfile='out1.ms'
        ct.split(vis='out3.ms',outputvis=msfile,datacolumn="data",antenna=ba,spw=spw,scan=goodsc)
        shutil.rmtree('out3.ms')
        rc=listobs(vis=msfile, listfile='obslist1.txt', verbose=True, overwrite=True)
        

        flagdata(msfile, mode='tfcrop')
        array=[]
        ms.open(msfile)
        for i in cal_scan:
            a=ms.statistics(column="data", complex_value='amplitude', scan=str(i),doquantiles=False)['']
            array.append([i,a['npts']])
        a=np.array(array); a= a[np.argsort( a[:,1])[::-1]]
        if flux_cal=='calc': flux_cal=int(a[0,0])
        
        if refant=='calc':
            array2=[]
            for i in range(len(goodoutant)):
                a=ms.statistics(column="data", complex_value='phase', scan=str(flux_cal),baseline=str(goodoutant[i]),doquantiles=False)['']
                array2.append([i,a['stddev'],a['npts']])
            a=np.array(array2)
            a= a[np.argsort( a[:,2])[::-1]]
            std=a[:,1]; ant=a[0,0]
            for i in range(5):
                if std[i+1]<std[i]:
                    ant=a[i+1,0]
                else:
                    break
            refant=goodoutant[int(ant)]
        ms.close()

    
        # metadata
        msmd.open(msfile)
        bcn = msmd.fieldsforscan(flux_cal, True)
        msmd.close()

        with open('filename',"a") as f:
            f.write(fr'# Observation no. {folder1}' +'\n'+
                    fr'# channel =   {channel}, bandwidth=   {bandwidth} MHz, channelwidth=   {chanwidth} kHz'+
                '\n'+f'# integration time=   {intime} sec, central frequency=   {round(ctrlfreq*1e-6)} MHz'+'\n'+
               f'# calibrator={cal_field}'+'\n'+fr'# Bad antenna   {badant}'+'\n'+
        f'# targets={tar_field}'+'\n'+
        fr'# flux calibrator is of field {bcn} with scan {int(flux_cal)}' +'\n'+
        fr'# Referance antenna is {refant}' +'\n')
        #shutil.copytree('out1.ms','out4.ms')

        cal_scan=list(set(cal_scan)-set(badscan))
        spw='0:'+str(int(0.5*channel))+'~'+str(int(.6*channel))
        pac=list(set(cal_scan)-set([int(flux_cal)]))
        pc=','.join(map(str, pac))
        for i in range(2):
            gaincal(msfile, caltable='phase.cal', scan=str(flux_cal), refant=refant, spw=spw,
                gaintype='G',calmode='p', solint='int',minsnr=5)
            gaincal(msfile,caltable='delay.cal', scan=str(flux_cal),refant=refant,spw=spw,gaintype='K', 
            solint='inf',minsnr=5,gaintable=['phase.cal'])
            bandpass(msfile,caltable='bandpass.cal',scan=str(flux_cal),spw='',refant=refant, minsnr=5,
                 solint='inf',bandtype='B',gaintable=['phase.cal','delay.cal'])
            gaincal(msfile,caltable='gain.cal',scan=str(flux_cal),spw=spw,minsnr=5,
                solint='int',refant=refant,gaintype='G',calmode='ap',solmode='R',
                gaintable=['delay.cal','bandpass.cal'])

            applycal(msfile, scan=str(flux_cal), gaintable=['delay.cal','gain.cal','bandpass.cal'],
                 calwt=False,parang=False)
            if len(pac)>0:
                gaincal(msfile,caltable='gain.cal',scan=pc,spw=spw,minsnr=5,
                solint='int',refant=refant,gaintype='G',calmode='ap',solmode='R',
                gaintable=['delay.cal','bandpass.cal'],interp=['linear',''],append=True)
                applycal(msfile,scan=pc, gaintable=['delay.cal','gain.cal','bandpass.cal'],
                 gainfield=['nearest','nearest',''],interp=['nearest','nearest',''],calwt=False,parang=False)
            flagdata(msfile, mode='tfcrop',datacolumn='corrected')
            flagdata(msfile, mode='rflag',datacolumn='corrected')
            for k in cal_scan:
                phaseflagging(msfile,k)
            shutil.rmtree('phase.cal'); shutil.rmtree('delay.cal');shutil.rmtree('gain.cal');shutil.rmtree('bandpass.cal')
        
        for i in cal_scan:
            baselflag(msfile,i,5,'corrected',1)
            ms.open(msfile)
            ms.select({'scan_number':[int(i)]})
            d1=ms.getdata(["flag"], ifraxis=True)
            visi=np.where(d1['flag']==False,1,np.nan)
            a1,a2,a3,a4=visi.shape
            antmean=[]
            for j in range(a3):
                antmean.append(np.nanmean(visi[:,:,j,:]))
            ms.close()
            if len(np.where(np.array(antmean)>0)[0])/len(antmean)<0.4:
                badscan.append(i)

        cal_scan=list(set(cal_scan)-set(badscan))
        tar_scan=list(set(tar_scan)-set(badscan))
        total_scans=list(set(total_scans)-set(badscan))
        goodsc=','.join(map(str, total_scans))

        if len(cal_scan)==0:
            os.chdir('/Data/jsalal/analysis/folder')
            shutil.rmtree(folder1)
            with open('pulsar.txt',"a") as f:
                f.write(fr'{folder1} # has all calibrator flagged' +'\n')
            return 1

        array2=[]
        ms.open(msfile)
        for i in range(len(goodoutant)):
            a=ms.statistics(column="corrected", complex_value='phase', scan=str(flux_cal),baseline=str(goodoutant[i]),doquantiles=False)['']
            array2.append([i,a['stddev'],a['npts']])
        ms.close()
        a=np.array(array2)
        a= a[np.argsort( a[:,2])[::-1]]
        std=a[:,1]; ant=a[0,0]
        for i in range(5):
            if std[i+1]<std[i]:
                ant=a[i+1,0]
            else:
                break
        refant=goodoutant[int(ant)]

        
        with open('filename',"a") as f:
            f.write(fr'# flagged scan {list(set(badscan))}' +'\n'+
                    fr'# Referance antenna is {refant}' +'\n')
        
        
        ct.split(vis='out1.ms',outputvis='out.ms',datacolumn="all",scan=goodsc)
        shutil.rmtree('out1.ms'); shutil.rmtree('out1.ms.flagversions')
        #shutil.rmtree('out4.ms')
    else:
        files = glob.glob("filename")
        lines=[]
        with open(files[0], encoding='utf8') as f:
            for line in f:
                lines.append(line)
        refant=lines[9][-4:-1]
                
    
    if os.path.isfile('bandpasscalibrator.pdf')==False:
        msfile='out.ms'
        spw='0:'+str(int(0.4*channel))+'~'+str(int(.6*channel))
        tr=','.join(map(str, tar_field))
        pac=list(set(cal_scan)-set([int(flux_cal)]))
        pc=','.join(map(str, pac))

        #  target flagging calibration
        for i in range(3):
            gaincal(msfile, caltable='phase.cal', scan=str(flux_cal), refant=refant, spw=spw,
            gaintype='G',calmode='p', solint='int',minsnr=5)
            gaincal(msfile,caltable='delay.cal', scan=str(flux_cal),refant=refant,spw=spw,gaintype='K', 
            solint='inf',minsnr=5,gaintable=['phase.cal'])
            bandpass(msfile,caltable='bandpass.cal',scan=str(flux_cal),spw='',refant=refant, minsnr=5,
                 solint='inf',bandtype='B',gaintable=['phase.cal','delay.cal'])
            gaincal(msfile,caltable='gain.cal',scan=str(flux_cal),spw=spw,minsnr=5,
                solint='int',refant=refant,gaintype='G',calmode='ap',solmode='R',
                gaintable=['delay.cal','bandpass.cal'])
            
            applycal(msfile, scan=str(flux_cal), gaintable=['delay.cal','gain.cal','bandpass.cal'],
                 calwt=False,parang=False)
            if len(pac)>0:
                gaincal(msfile,caltable='gain.cal',scan=pc,spw=spw,minsnr=5,
                solint='int',refant=refant,gaintype='G',calmode='ap',solmode='R',
                gaintable=['delay.cal','bandpass.cal'],interp=['linear',''],append=True)
                applycal(msfile,scan=pc, gaintable=['delay.cal','gain.cal','bandpass.cal'],
                 gainfield=['nearest','nearest',''],interp=['nearest','nearest',''],calwt=False,parang=False)
            applycal(msfile,field=tr,gaintable=['delay.cal','gain.cal','bandpass.cal'],
                 gainfield=['nearest','nearest',''], interp=['nearest','nearest',''],
                 calwt=False,parang=False)
            flagdata(msfile, mode='tfcrop',datacolumn='corrected')
            flagdata(msfile, mode='rflag',datacolumn='corrected')
            phaseflagging(msfile,str(flux_cal))
            for k in pac:
                phaseflagging(msfile,k)
            if i>1:flagdata(msfile, mode='extend',datacolumn='corrected',growtime=80.0, growfreq=80.0,growaround=True)
            shutil.rmtree('phase.cal'); shutil.rmtree('delay.cal');shutil.rmtree('gain.cal');shutil.rmtree('bandpass.cal')
            
            if i==1:
                msmd.open(msfile)
                antenna_names=msmd.antennanames()
                msmd.close()
                flagged_antenna=flagant(msfile,cal_scan,channel,antenna_names)
                baselr=baseline(msfile,badant=flagged_antenna[0])
                flagged_base=flagant(msfile,cal_scan,channel,baselr)
                badant=flagged_antenna[0]; badbase=flagged_base[0]
                baselr=baseline(msfile,badant,badbase)
                ba=';'.join(map(str, baselr))
                ct.split(msfile,outputvis='out2.ms',datacolumn="corrected",antenna=ba)
                shutil.rmtree(msfile); os.rename('out2.ms',msfile)
                for i in range(len(flagged_antenna[2])):
                    ba=';'.join(map(str, flagged_antenna[1][i]))
                    if flagged_antenna[2][i] - 1 in tar_scan:
                        flagdata(msfile,mode="manual", antenna=ba,scan=str(flagged_antenna[2][i]-1))
                    if flagged_antenna[2][i] + 1 in tar_scan:
                        flagdata(msfile,mode="manual", antenna=ba,scan=str(flagged_antenna[2][i]+1))
                for i in range(len(flagged_base[2])):
                    ba=';'.join(map(str, flagged_base[1][i]))
                    if flagged_base[2][i] - 1 in tar_scan:
                        flagdata(msfile,mode="manual", antenna=ba,scan=str(flagged_base[2][i]-1))
                    if flagged_base[2][i] + 1 in tar_scan:
                        flagdata(msfile,mode="manual", antenna=ba,scan=str(flagged_base[2][i]+1))
        for k in cal_scan:
            plotms(msfile,plotfile='2atc'+str(k)+'.png',scan=str(k),ydatacolumn='corrected',
               showgui=False,highres=False,width=600,height=350,overwrite=True)
            plotms(msfile,plotfile='2afc'+str(k)+'.png',xaxis='frequency',scan=str(k),ydatacolumn='corrected',
               showgui=False,highres=False,width=600,height=350,overwrite=True)
            plotms(msfile,plotfile='2uac'+str(k)+'.png',xaxis='UVwave',scan=str(k),ydatacolumn='corrected',
               showgui=False,highres=False,width=600,height=350,overwrite=True)
            plotms(msfile,plotfile='2pac'+str(k)+'.png',xaxis='Phase',scan=str(k),xdatacolumn='corrected',ydatacolumn='corrected',
               showgui=False,highres=False,width=600,height=350,overwrite=True)
            file=glob.glob('*'+str(k)+'.png')
            imagl=[]
            for i in file:
                a1=Image.open(i)
                imagl.append(a1.convert('RGB'))
            if k==int(flux_cal):
                imagl[0].save('bandpasscalibrator.pdf',save_all=True, append_images=imagl[1:])
            else:
                imagl[0].save('calibrator'+str(k)+'.pdf',save_all=True, append_images=imagl[1:])
            for a in file: os.remove(a)
        
            
        b=np.array(['#sourceid','RA','DEC','isnr','flux','dsnoise','PPR',
                   'sfreq','esfreq','ampf','eampf','stime','estime','ampt','eampt','modulation','csnr','ctrlfreq','bandwidth','chanwidth','ttime','subint','MJD','sample'])
        with open('filename','ab') as f:
            np.savetxt(f, b, fmt='%18s',delimiter=',', newline='')
            f.write(b'\n')
        #shutil.rmtree('out2.ms'); shutil.rmtree('out2.ms.flagversions')
        shutil.rmtree('out.ms.flagversions')

    # 3. Process Target Fields
    for i in tar_field:
        msmd.open(msfile)
        cscan = list(msmd.scansforfield(i))
        ctrlfreq = msmd.meanfreq(0)
        msmd.close()
        
        a = 1.22 * 3e8 / ctrlfreq / 25e3 * u.rad
        res = round((a.to(u.arcsec) / 5).value, 1)
        
        for j in cscan:
            scan_dir = os.path.join(target_dir, str(j))
            if not os.path.isdir(scan_dir):
                os.mkdir(scan_dir)
            os.chdir(scan_dir)
            
            # Call the extracted imaging and extraction function
            process_target_field(
                field=i, scan=str(j), res=res, refant=refant, base_dir=base_dir, 
                dire=target_dir, channel=channel, chanwidth=chanwidth, 
                intime=intime, ctrlfreq=ctrlfreq, bandwidth=bandwidth
            )
            os.chdir(target_dir)

    # 4. Cleanup
    if os.path.isdir("out.ms"):
        shutil.rmtree("out.ms")
        
    if send:
        cleanup_files = glob.glob("*S.fits") + glob.glob("*.plan") + glob.glob("*.lta")
        for file in cleanup_files: 
            if os.path.exists(file): os.remove(file)
            
    end = time.time()
    
    # FIXME: 'pcan' and 'sourcedec' tracking should be implemented during the process_target_field loop.
    pcan = 0 
    sourcedec = []
    
    os.chdir(base_dir)
    success_msg = f'{folder} # successfully analysed. Found {pcan} pulsar candidates from {len(sourcedec)} scintillators. Total time = {round((end - start)/3600, 2)} hrs\n'
    with open('pulsar_summary.txt', "a") as f:
        f.write(success_msg)
        
    print(success_msg.strip())
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main execution script SPICE.")
    parser.add_argument("--target_dir", type=str, required=True, help="Target observation to process.")
    parser.add_argument("--flux_cal", type=str, default="calc", help="Flux calibrator override.")
    parser.add_argument("--refant", type=str, default="calc", help="Reference antenna override.")
    
    args = parser.parse_args()
    
    try:
        data_analysis(
            target_dir=args.target_dir,
            flux_cal=args.flux_cal,
            refant=args.refant
        )
    except Exception as e:
        print(f"SPICE failed: {e}")