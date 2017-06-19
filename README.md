
# README.md : Tutorial to use the python tools for atmospheric analysis
===========================================================================

- author : Sylvie Dagoret-Campagne
- affiliation : LAL/IN2P3/CNRS
- date June 19th 2017
- 

These tools are the analysis too for the CTIO June data.


## 1) View the raw images
----------------------------------

- ViewAllImages.ipynb

## 2) Reduce the images
---------------------------------

The images are reduced according a computed Master Bias and Master Flat.

- ReduceAllImages.ipynb


## 3) Make a logbook
----------------------
- MakeLogBook.ipynb

gather usefull information in a single logbook which is used to simulate the spectra


## 4) Find the central star
--------------------------------

Find where is the target star to extract a subimage relevant for extracting the spectra

- FindCentralStar.ipynb  : bad not used
- FindCentralStarFast.ipynb : very simple localiztion of the main star in a specified subimage, should be used


## 5) Find Rotation of disperser axis wrt CCD
------------------------------------------------------------

- FindOptRot.ipynb

Notice jeremy has addeda method basedon hessian that works well



## 6) Extract Spectrum
------------------------

Extract the region whre the spectrum is. Project the spectumalong a 1-D array.

- Extract_Spectrum.ipynb	


## 7) Calibrate the spectra
-------------------------------

- calibrate from pixel to wavelength 

## 8) Generate the simulated spectra
------------------------------------

- GenerateSimulationProfiles.ipynb


## 9) Compare data and sim spectra
------------------------------------------------

- CalibrateSpectrum.ipynb		


## 10) Compute aerosols attenuation
---------------------------------------------

- AnaAerCalibSpectrum.ipynb		

## 11) Compute Equivalent width
--------------------------------

- AnaEqWdtCalibSpectrum.ipynb		
	
		

# Other tools
-----------------


## Overscan and Trim

- subtract Overscan and Trim the images

## Compute Master Bias

- BuildMasterBias

## Compute Master Flat

- BuildMasterFlat




