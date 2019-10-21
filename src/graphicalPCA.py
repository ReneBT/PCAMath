import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
#from matplotlib.patches import ConnectionPatch
#from matplotlib import transforms
from sklearn.decomposition import PCA

import src.local_nipals as npls

# This expects to be called inside the jupyter project folder structure.
from src.file_locations import data_folder,images_folder


# Using pathlib we can handle different filesystems (mac, linux, windows) using a common syntax.
# file_path = data_folder / "raw_data.txt"
# More info on using pathlib:
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


class graphicalPCA:
### ******      START CLASS      ******
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

    def __init__(pca_full_covariance):
### ***   START  Data Calculations   *** 
### Read in data
# simulated fatty acid spectra and associated experimental concentration data
        GC_data = sio.loadmat(
            data_folder / "FA profile data.jrb", struct_as_record=False
        )
# Gas Chromatograph (GC) data is from Beattie et al. Lipids 2004 Vol 39 (9):897-906
        simplified_fatty_acid_spectra = sio.loadmat(data_folder / "FA spectra.mat", struct_as_record=False)
# simplified_fatty_acid_spectra are simplified spectra of fatty acid methyl esters built from the properties described in
# Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        wavelength_axis = simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,]
        min_spectral_values = np.tile(
            np.min(simplified_fatty_acid_spectra["simFA"], axis=1), (np.shape(simplified_fatty_acid_spectra["simFA"])[1], 1)
        )  
# convert the mass base profile provided into a molar profile
        molar_profile = GC_data["ANNSBUTTERmass"] / simplified_fatty_acid_spectra["FAproperties"][0, 0].MolarMass
        molar_profile = 100.0 * molar_profile / sum(molar_profile)

### generate simulated observational
# spectra for each sample by multiplying the simulated FA reference spectra by 
# the Fatty Acid profiles. Note that the simplified_fatty_acid_spectra spectra 
# have a standard intensity in the carbonyl mode peak (the peak with the 
# highest pixel position)
        spectra_full_covariance = np.dot(simplified_fatty_acid_spectra["simFA"], molar_profile)
        min_spectra_full_covariance = np.dot(
            np.transpose(min_spectral_values), molar_profile
        )  # will allow scaling of min_spectral_values to individual sample

### calculate PCA
# with the full fatty acid covariance, comparing custom NIPALS and built in PCA function.

        pca_full_covariance = npls.nipals(
            X_data=spectra_full_covariance,
            maximum_number_PCs=13,
            maximum_iterations_PCs=10,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_spectra_full_covariance,
        )
        pca_full_covariance.calc_PCA()

        pca_full_covariance_builtin = PCA(
            n_components=51
        )  # It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        pca_full_covariance_builtin.fit(np.transpose(spectra_full_covariance))

### Plot Pure Simulated Data
        plt.plot( wavelength_axis , simplified_fatty_acid_spectra["simFA"] )
        plt.ylabel("Intensity / Counts per 100 Counts Carbonyl Mode")
        plt.xlabel("Raman Shift cm$^{-1}$")
        image_name = " Pure Fatty Acid Simulated Spectra."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)        # plt.show()
        plt.close()
        
### Plot Simulated Observational Data        
        plt.plot( wavelength_axis , spectra_full_covariance )
        plt.ylabel("Intensity / Counts per 100 Counts Carbonyl Mode")
        plt.xlabel("Raman Shift cm$^{-1}$")
        image_name = " All Simulated Observations Data Plot."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)        # plt.show()
        plt.close()

### convergence of PCs 
# between NIPALS and SVD based on tolerance
        PCFulldiffTol = np.empty([15, 50])
        for d in range(14):
            tempNIPALS = npls.nipals(
                spectra_full_covariance, 51, 20, 10 ** -(d + 1), "MC"
            )  # d starts at 0
            tempNIPALS.calc_PCA()
            for iPC in range(49):
                PCFulldiffTol[d, iPC] = np.log10(
                    np.minimum(
                        np.mean(
                            np.absolute(
                                tempNIPALS.REigenvector[iPC,]
                                - pca_full_covariance_builtin.components_[iPC,]
                            )
                        ),
                        np.mean(
                            np.absolute(
                                tempNIPALS.REigenvector[iPC,]
                                + pca_full_covariance_builtin.components_[iPC,]
                            )
                        ),
                    )
                )  # capture cases of inverted REigenvectors (arbitrary sign switching)

### compare NIPALs output to builtin SVD output 
# for 1st PC, switching sign if necessary as this is arbitrary
        if np.sum(
            np.absolute(tempNIPALS.REigenvector[0,] - pca_full_covariance_builtin.components_[0,])
        ) < np.sum(
            np.absolute(
                tempNIPALS.REigenvector[iPC,] + pca_full_covariance_builtin.components_[iPC,]
            )
        ):
            sign_Switch = 1
        else:
            sign_Switch = -1
        plt.plot( pca_full_covariance.pixel_axis, 
                 pca_full_covariance.REigenvector[0,],
                 pca_full_covariance.pixel_axis, 
                 sign_Switch*pca_full_covariance_builtin.components_[0,], ":"
        )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        plt.gca().legend(["NIPALS","SVD"])
        image_name = " Local NIPALS overlaid SKLearn SVD based Principal Components."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)        # plt.show()
        plt.close()

        # NIPALS minus builtin SVD based output for 1st PC
        plt.plot(
            pca_full_covariance.pixel_axis,
            pca_full_covariance.REigenvector[0,] - 
            sign_Switch*pca_full_covariance_builtin.components_[0,],
        )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        image_name = " Local NIPALS minus SKLearn SVD based Principal Components."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)
        plt.close()

        # plot the change in difference between NIPALS and SVD against tolerance for PC1
        plt.plot(list(range(-1, -15, -1)), PCFulldiffTol[0:14, 3])
        plt.ylabel("$Log_{10}$ of the Mean Absolute Difference")
        plt.xlabel("$Log_{10}$ NIPALS Tolerance")
        image_name = " Local NIPALS minus SKLearn SVD vs Tolerance."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)
        plt.close()

        # plot the change in difference between NIPALS and SVD against PC rank for tolerance of 0.001
        plt.plot(
            list(range(1, 49)),
            PCFulldiffTol[0, 0:48],
            list(range(1, 49)),
            PCFulldiffTol[11, 0:48],

        )
        plt.ylabel("$Log_{10}$ of the Mean Absolute Difference")
        plt.xlabel("PC Rank")
        plt.gca().legend(["Tol = 0.1","Tol=10$^{-12}$"])
        image_name = " Local NIPALS minus SKLearn SVD vs Rank."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)
        plt.close()
        
        corrPCs = np.inner(pca_full_covariance_builtin.components_,
                           tempNIPALS.REigenvector)
        #loadings already standardised to unit norm
        maxCorr = np.max(np.abs(corrPCs),axis=0)
        max_Ix = np.empty([51,1])
        for iCol in range(np.shape(maxCorr)[0]):
            max_Ix[iCol] = np.where(np.abs(corrPCs[:,iCol])==maxCorr[iCol])

        plt.plot(
            list(range(1, 52)),
            max_Ix,
            "."
        )
        plt.ylabel("SVD rank")
        plt.xlabel("NIPALS rank")
        image_name = " Local NIPALS Rank vs SKLearn SVD Rank."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)
        plt.close()
        
        plt.plot(
            list(range(1, 52)),
            maxCorr,
            ".",markersize=10,)
        plt.plot(        
            list(range(1, 52)),
            np.abs(np.diag(corrPCs)),
            ".",markersize=5,
        )
        plt.ylabel("SVD/NIPALS Correlation")
        plt.xlabel("NIPALS rank")
        plt.legend(["Optimum Correlation","Rank Correlation"])
        image_name = " Correlation between SVD and NIPALs by closest identity and by rank."
        full_path = os.path.join(images_folder, pca_full_covariance.fig_Project +
                                image_name + pca_full_covariance.fig_Format)
        plt.savefig(full_path, 
                         dpi=pca_full_covariance.fig_Resolution)
        plt.close()
        
### ***   END  Data Calculations   ***


### ***   START Equations   ***
### Matrix Equations
# Generate FIGURE for the main PCA equation in its possible arrangements
# Row data matrix (spectra displayed as rows by convention)
        pca_full_covariance.figure_DSLT("DSLT")
        pca_full_covariance.figure_DSLT("SDL")
        pca_full_covariance.figure_DSLT("LTSiD")
### Vector Equations
        iPC = 1
        pca_full_covariance.figure_sldi(iPC)
        pca_full_covariance.figure_lsdi(iPC)
### Algorithm Equations
# Function to generate sum of squares figure showing how PCA is initialised
        pca_full_covariance.figure_DTD()
        pca_full_covariance.figure_DTDw()
### ***   END Equations   ***

### ***   START Common Signal   ***
# scaling factor calculated for subtracting the common signal from the positive
# and negative constituents of a PC. Use non-mean centred data for clarity
        nPC = 13
        pca_full_covarianceNMC = npls.nipals(
            X_data = spectra_full_covariance,
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 10,
            iteration_tolerance = 0.000000000001,
            preproc = "NA",
            pixel_axis = wavelength_axis,
            spectral_weights = molar_profile,
            min_spectral_values = min_spectra_full_covariance,
        )
        pca_full_covarianceNMC.calc_PCA()
# generate subtraction figures for positive vs negative score weighted sums.
        pca_full_covarianceNMC.calc_Constituents(nPC)
        xview = range(625, 685)  # zoom in on region to check in detail for changes
        iPC = 2 #PC1 is not interesting for common signal (there is none, so have iPC>=2)
        pca_full_covarianceNMC.figure_lpniLEigenvectorEqn(iPC)
# generate overlays of the constituents at adjusted scaling factors
        pca_full_covarianceNMC.figure_lpniCommonSignalScalingFactors(nPC, xview)
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC)
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC, pca_full_covarianceNMC.optSF[iPC-1]*1.01)#SF 1% larger
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC, pca_full_covarianceNMC.optSF[iPC-1]*0.99)#Sf 1% smaller
        
### ***   END  Common Signal   ***
### ******      END CLASS      ******


        return
