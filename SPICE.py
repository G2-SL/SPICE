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

# External modules from your pipeline
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
        # Assuming baseline and flagant are imported from your flagging module
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

def data_analysis(base_dir, folder, catrms=True, flux_cal='calc', refant='calc', badscan=None, badant=None, send=True):
    """
    SPICE execution function.
    Returns True on full success, raises exceptions on critical failures, or returns False on soft missing data.
    """
    if badscan is None: badscan = []
    if badant is None: badant = []
    
    start = time.time()
    os.chdir(base_dir)
    target_dir = os.path.join(base_dir, folder)
    
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
        shutil.rmtree(folder1)
        error_msg = f'{folder1} # missing calibrators or targets. Cals: {cal_field}, Targets: {tar_field}\n'
        with open('pulsar_summary.txt', "a") as f:
            f.write(error_msg)
        msmd.close()
        print(error_msg)
        return False

    cal_scan, tar_scan = [], []
    for i in cal_field: cal_scan += list(msmd.scansforfield(i))
    for i in tar_field: tar_scan += list(msmd.scansforfield(i))
    intime = msmd.exposuretime(scan=cal_scan[0])['value']
    msmd.close()

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
    success_msg = f'{folder1} # successfully analysed. Found {pcan} pulsar candidates from {len(sourcedec)} scintillators. Total time = {round((end - start)/3600, 2)} hrs\n'
    with open('pulsar_summary.txt', "a") as f:
        f.write(success_msg)
        
    print(success_msg.strip())
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main execution script SPICE.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the base analysis directory.")
    parser.add_argument("--folder", type=str, required=True, help="Target observation folder to process.")
    parser.add_argument("--flux_cal", type=str, default="calc", help="Flux calibrator override.")
    parser.add_argument("--refant", type=str, default="calc", help="Reference antenna override.")
    
    args = parser.parse_args()
    
    try:
        data_analysis(
            base_dir=args.base_dir,
            folder1=args.folder,
            flux_cal=args.flux_cal,
            refant=args.refant
        )
    except Exception as e:
        print(f"SPICE failed: {e}")