import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import transforms
from sklearn.decomposition import PCA

import src.local_nipals as npls

# This expects to be called inside the jupyter project folder structure.
from src.file_locations import data_folder,images_folder


# Using pathlib we can handle different filesystems (mac, linux, windows) using a common syntax.
# file_path = data_folder / "raw_data.txt"
# More info on using pathlib:
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


class graphicalPCA:
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

    def __init__(self):
###################            START Data Simulation              #######################
### Read in simulated fatty acid spectra and associated experimental concentration data
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

# Now generate simulated spectra for each sample by multiplying the simulated FA reference
# spectra by the Fatty Acid profiles. Note that the simplified_fatty_acid_spectra spectra have a standard intensity in the
# carbonyl mode peak (the peak with the highest pixel position)
        spectra_full_covariance = np.dot(simplified_fatty_acid_spectra["simFA"], molar_profile)
        min_spectra_full_covariance = np.dot(
            np.transpose(min_spectral_values), molar_profile
        )  # will allow scaling of min_spectral_values to individual sample

        plt.plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], spectra_full_covariance)
        plt.ylabel("Intensity")
        plt.xlabel("Raman Shift cm$^-1$")
        plt.savefig(images_folder / "Simulated Data full covariance.png", dpi=300)
        plt.close()

# Now we calculate PCA, with the full fatty acid covariance, comparing custom NIPALS and built in PCA function.

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

        # test convergence of PCs between NIPALS and SVD based on tolerance
        PCFulldiffTol = np.empty([15, 50])
#        PC2diffTol = np.empty([15, 50])
        for d in range(14):
            tempNIPALS = npls.nipals(
                spectra_full_covariance, 51, 20, 10 ** -(d + 1), "MC"
            )  # d starts at 0
            tempNIPALS.calc_PCA()
            for iPC in range(49):
                PCFulldiffTol[d, iPC] = np.log10(
                    np.minimum(
                        np.sum(
                            np.absolute(
                                tempNIPALS.REigenvector[iPC,]
                                - pca_full_covariance_builtin.components_[iPC,]
                            )
                        ),
                        np.sum(
                            np.absolute(
                                tempNIPALS.REigenvector[iPC,]
                                + pca_full_covariance_builtin.components_[iPC,]
                            )
                        ),
                    )
                )  # capture cases of inverted REigenvectors (arbitrary sign switching)

        # compare NIPALs output to builtin SVD based output for 1st PC, switching sign if necessary as this is arbitrary
        plt.plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.REigenvector[0,])
        if np.sum(
            np.absolute(tempNIPALS.REigenvector[0,] - pca_full_covariance_builtin.components_[0,])
        ) < np.sum(
            np.absolute(
                tempNIPALS.REigenvector[iPC,] + pca_full_covariance_builtin.components_[iPC,]
            )
        ):
            plt.plot(
                simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance_builtin.components_[0,], ":"
            )
        else:
            plt.plot(
                simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], -pca_full_covariance_builtin.components_[0,], ":"
            )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        # plt.show()
        plt.close()

        # NIPALS minus builtin SVD based output for 1st PC
        plt.plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.REigenvector[0,] + pca_full_covariance_builtin.components_[0,],
        )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        # plt.show()
        plt.close()

        # plot the change in difference between NIPALS and SVD against tolerance for PC1
        plt.plot(list(range(-1, -15, -1)), PCFulldiffTol[0:14, 3])
        plt.ylabel("$Log_{10}$ of the Absolute Sum of Difference")
        plt.xlabel("$Log_{10}$ NIPALS Tolerance")
        # plt.show()
        plt.close()

        # plot the change in difference between NIPALS and SVD against PC rank for tolerance of 0.001
        plt.plot(
            list(range(1, 49)),
            PCFulldiffTol[0, 0:48],
            list(range(1, 49)),
            PCFulldiffTol[1, 0:48],

        )
        plt.ylabel("$Log_{10}$ of the Absolute Sum of Difference")
        plt.xlabel("PC Rank")
        plt.close()

###################         END  Data Calculations            #######################


###################            START Matrix Equations              #######################
# Generate FIGURE for the main PCA equation in its possible arrangements
        # Row data matrix (spectra displayed as rows by convention
        pca_full_covariance.figure_DSLT("DSLT")
        pca_full_covariance.figure_DSLT("SDL")
        pca_full_covariance.figure_DSLT("LTSiD")
###################             END  Matrix Equations             #######################

###################            START Vector Equations             #######################
        iPC = 1
        pca_full_covariance.figure_sldi(iPC)
        pca_full_covariance.figure_lsdi(iPC)
###################             END  Vector Equations             #######################


###################            START Algorithm Equations             #######################
# Function to generate sum of squares figure showing how PCA is initialised
        pca_full_covariance.figure_DTD()
        pca_full_covariance.figure_DTDw()

### NOTE THIS CODE IS CURRENTLY RETAINED TO PREVENT BREAKING DOWNSTREAM FIGURE PLOTS UNTIL ALL ARE CONVERTED TO CLASS FUNCTIONS ####
        ###################       START lpniCommonSignalScalingFactors       #######################
        # FIGURE of the scaling factor calculated for subtracting the common signal from the positive
        # and negative constituents of a PC
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
        ###################         END lpniCommonSignalScalingFactors           #######################
###################            START Positive Negative Equations             #######################
# generate subtraction figures for positive vs negative score weighted sums.
        pca_full_covarianceNMC.calc_Constituents(nPC)
        xview = range(625, 685)  # zoom in on region to check in detail for changes
        iPC = 2 #PC1 is not interesting for common signal (there is none, so have iPC>=2)
        pca_full_covarianceNMC.figure_lpniLEigenvectorEqn(iPC)

        pca_full_covarianceNMC.figure_lpniCommonSignalScalingFactors(nPC, xview)
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC)
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC, pca_full_covarianceNMC.optSF[iPC-1]*1.01)#SF 1% larger
        pca_full_covarianceNMC.figure_lpniCommonSignal(iPC, pca_full_covarianceNMC.optSF[iPC-1]*0.99)#Sf 1% smaller
        
###################             END  Positive Negative Equations             #######################


        return
