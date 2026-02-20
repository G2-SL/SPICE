# SPICE: Scintillation Pipeline for Interferometric Candidate Extraction

**SPICE** is an automated, CASA-based data reduction and analysis pipeline designed to identify pulsar candidates in Giant Metrewave Radio Telescope (GMRT) data by detecting their Diffractive Interstellar Scintillation (DISS) signatures.



Pulsars are point-like sources that exhibit high modulation in flux across time and frequency due to the ionized interstellar medium. SPICE automates the transition from raw interferometric data to scintillation parameters, allowing for scintillating candidate extraction.

## Features

* **End-to-End Automation:** Converts raw GMRT data (`.lta`, `.fits`) into calibrated Measurement Sets (MS).
* **Precision Flagging:** Multi-stage flagging (time, channel, and baseline) to mitigate terrestrial RFI.
* **Blind Source Extraction:** Integrated with **PyBDSF** to identify high-SNR point sources in continuum images.
* **Scintillation Modeling:** * Generates dynamic spectra from calibrated visibilities.
    * Computes 2D autocorrelations to determine scintillation bandwidth ($\Delta\nu_d$) and timescale ($\Delta t_d$).
    * Performs 2D Gaussian fitting to extract DISS parameters and modulation indices.
* **Diagnostic Visualization:** Generates plots for every candidate, showing dynamic spectra alongside their ACF fits.

## Pipeline Architecture



The pipeline is modularized into three core components:
1. `Flagging.py`: RFI mitigation and baseline quality control.
2. `Scintillation.py`: Dynamic spectra extraction and 2D Gaussian modeling.
3. `SPICE.py`: The main pipeline that manages calibration, imaging, and candidate logging.

## Installation & Requirements

### Prerequisites
* **CASA (Common Astronomy Software Applications):** Version 6.2 or higher is recommended.
* **Python 3.10.12:** Available within the CASA environment.

### Dependencies
Install the required Python libraries via pip:
```bash
pip install -r requirements.txt
 ```
## Quick Start

To begin extracting pulsar candidates, follow these steps:

1.  **Prepare your environment:** Ensure your observation folder contains the raw data (`.lta`, `.fits`, or `.ms`).
2.  **Calibrator List:** Place a file named `cals.list` in the parent directory. This file should contain the names of known GMRT/VLA calibrators used in your observation.
3.  **Run the pipeline:**
    Open your terminal and execute the SPICE script:
    ```bash
    python SPICE.py --target_dir ./path/to/observation_folder
    ```

### Optional Arguments
You can customize the calibration behavior using the following flags:
* `--flux_cal`: Manually specify the scan number for the flux calibrator (e.g., `--flux_cal 4`).
* `--refant`: Manually specify the reference antenna (e.g., `--refant C02`).

---

## Pipeline Outputs

SPICE generates structured data and visual diagnostics to help you quickly verify pulsar candidates:

### 1. Tabular Data & Logs
* **`scintillation_log.csv`**: The primary science output. A catalog containing RA, DEC, SNR, scintillation bandwidth ($\Delta\nu_d$), and timescale ($\Delta t_d$).
* **`pulsar_summary.txt`**: A high-level summary tracking the number of candidates found and total processing time.
* **`pipeline_log.txt`**: Detailed technical logs including flagging percentages and RMS statistics from PyBDSF.

### 2. Diagnostic Plots
The pipeline produces **`source_X.jpg`** files for every identified candidate. These plots include:
* The raw and polynomial-fitted dynamic spectra for both polarizations.
* The 2D autocorrelation function (ACF).
* 1D slices of the ACF with the corresponding Gaussian fits to visualize the scintillation parameters.



---

## Contributing & Research
This pipeline was developed as part of a PhD thesis to streamline the identification of pulsars via their diffractive interstellar scintillation signatures. 

If you use **SPICE** in your research, please cite this repository. For bugs or feature requests, please open an issue in the GitHub tracker.
