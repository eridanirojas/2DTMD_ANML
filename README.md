# 2DTMD_ANML
Analysis toolkit for Raman characterization of two-dimensional transition metal dichalcogenides (2D-TMDs) developed for the Advanced Nanomaterials and Manufacturing Lab.

This repository contains tools for processing Raman spectroscopy data, extracting spectral features, visualizing spatial uniformity across surfaces, and training machine learning models linking CVD process parameters to thin-film quality.

---

## ⚙️ Installation

Run the following in your terminal:

# bash/zsh
`git clone git@github.com:eridanirojas/2DTMD_ANML.git`

# navigate into the repository
`cd 2DTMD_ANML`

# Create the conda environment
`conda env create -f environment.yml`

# Activate the environment
`conda activate 2dtmd_anml`

# Install the package locally
`pip install -e .`

---

## 📂 Repository Structure

### src/

This directory contains the core analysis modules used for Raman spectroscopy processing and feature extraction.

- **processing.py**  
  Toolkit for preparing Raman spectra:
  - reads data directly from the Raman tool
  - baseline subtraction
  - normalization of spectral intensity

- **peaks.py**  
  Peak detection algorithm that extracts spectral features including:
  - peak location
  - FWHM (full width at half maximum)
  - peak area
  - peak-to-peak separation
  - peak-to-peak intensity ratios

- **fitting.py**  
  Spectral peak fitting using:
  - Gaussian models
  - Lorentzian models
  - Voigt models

- **visualization.py**  
  Visualization tools for communicating extracted spectral features.  
  Includes functionality to iterate through a Raman map scan and generate heatmaps across a surface based on selected peak-to-peak ratios.

---

### training/

Machine learning workflows linking **CVD process parameters to thin-film quality metrics**.

- **cvdRecipeChewing.ipynb**  
  Processes CVD recipe files so that the parameters can be used effectively for machine learning models.

- **training.ipynb**  
  Implements and trains a TensorFlow neural network regression model to predict thin-film quality metrics from CVD process parameters.

---

### example/

Example workflows demonstrating the capabilities of the repository.

- **raman_analysis.ipynb**  
  Full demonstration notebook showing how to:
  - preprocess Raman spectra
  - detect and fit peaks
  - extract spectral features
  - visualize material quality metrics

---

### xps/

Early stage work intended for future development of XPS analysis tools.  
This portion of the project was not completed but represents a planned extension of the Raman analysis framework.

---

## 🔬 Typical Analysis Workflow

A typical workflow using this toolkit involves:

1. Import Raman spectra from the instrument
2. Remove baseline and normalize spectra
3. Detect spectral peaks
4. Fit peaks using Gaussian/Lorentzian/Voigt models
5. Extract spectral features such as FWHM and peak ratios
6. Visualize spatial variation across map scans
7. Optionally train ML models linking process parameters to material quality

See `example/raman_analysis.ipynb` for a full demonstration.
