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

    def __init__(pcaMC):
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
        data = np.dot(simplified_fatty_acid_spectra["simFA"], molar_profile)
        min_data = np.dot(
            np.transpose(min_spectral_values), molar_profile
        )  # will allow scaling of min_spectral_values to individual sample

### calculate PCA
# with the full fatty acid covariance, comparing custom NIPALS and built in PCA function.

        pcaMC = npls.nipals(
            X_data=data,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pcaMC.calc_PCA()

        pcaMC_builtin = PCA(
            n_components=51
        )  # It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        pcaMC_builtin.fit(np.transpose(data))

### Plot Pure Simulated Data
        plt.plot( wavelength_axis , simplified_fatty_acid_spectra["simFA"] )
        plt.ylabel("Intensity / Counts per 100 Counts Carbonyl Mode")
        plt.xlabel("Raman Shift cm$^{-1}$")
        image_name = " Pure Fatty Acid Simulated Spectra."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)        # plt.show()
        plt.close()
        
### Plot Simulated Observational Data        
        plt.plot( wavelength_axis , data )
        plt.ylabel("Intensity / Counts per 100 Counts Carbonyl Mode")
        plt.xlabel("Raman Shift cm$^{-1}$")
        image_name = " All Simulated Observations Data Plot."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)        # plt.show()
        plt.close()

### convergence of PCs 
# between NIPALS and SVD based on tolerance
        PCFulldiffTol = np.empty([15, 50])
        for d in range(14):
            tempNIPALS = npls.nipals(
                data, 51, 100, 10 ** -(d + 1), "MC"
            )  # d starts at 0
            tempNIPALS.calc_PCA()
            for component_index in range(49):
                PCFulldiffTol[d, component_index] = np.log10(
                    np.minimum(
                        np.mean(
                            np.absolute(
                                tempNIPALS.spectral_loading[component_index,]
                                - pcaMC_builtin.components_[component_index,]
                            )
                        ),
                        np.mean(
                            np.absolute(
                                tempNIPALS.spectral_loading[component_index,]
                                + pcaMC_builtin.components_[component_index,]
                            )
                        ),
                    )
                )  # capture cases of inverted Loadings (arbitrary sign switching)

### compare NIPALs output to builtin SVD output 
# for 1st PC, switching sign if necessary as this is arbitrary
        if np.sum(
            np.absolute(tempNIPALS.spectral_loading[0,] - pcaMC_builtin.components_[0,])
        ) < np.sum(
            np.absolute(
                tempNIPALS.spectral_loading[component_index,] + pcaMC_builtin.components_[component_index,]
            )
        ):
            sign_Switch = 1
        else:
            sign_Switch = -1
        plt.plot( pcaMC.pixel_axis, 
                 pcaMC.spectral_loading[0,],
                 pcaMC.pixel_axis, 
                 sign_Switch*pcaMC_builtin.components_[0,], ":"
        )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        plt.gca().legend(["NIPALS","SVD"])
        image_name = " Local NIPALS overlaid SKLearn SVD based Principal Components."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)        # plt.show()
        plt.close()

        # NIPALS minus builtin SVD based output for 1st PC
        plt.plot(
            pcaMC.pixel_axis,
            pcaMC.spectral_loading[0,] - 
            sign_Switch*pcaMC_builtin.components_[0,],
        )
        plt.ylabel("Weights")
        plt.xlabel("Raman Shift")
        image_name = " Local NIPALS minus SKLearn SVD based Principal Components."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()

        # plot the change in difference between NIPALS and SVD against tolerance for PC1
        plt.plot(list(range(-1, -15, -1)), PCFulldiffTol[0:14, 3])
        plt.ylabel("$Log_{10}$ of the Mean Absolute Difference")
        plt.xlabel("$Log_{10}$ NIPALS Tolerance")
        image_name = " Local NIPALS minus SKLearn SVD vs Tolerance."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
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
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
        
        corrPCs = np.inner(pcaMC_builtin.components_,
                           tempNIPALS.spectral_loading)
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
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
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
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
        
        plt.plot( wavelength_axis,
                 data-
                     (np.inner(pcaMC.component_weight,
                           pcaMC.spectral_loading.T)
                     +pcaMC.centring
                     ).T
                 )
        plt.ylabel("Residual Signal")
        plt.xlabel("Raman Shift /cm$^{-1}$")
        image_name = " Difference between data and SLT reconstructed data."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
        
        
        plt.plot( (pcaMC.component_weight[:,:12]-
                     (np.inner(data.T-pcaMC.centring,
                           pcaMC.spectral_loading[:12,:])      
                     )).T
                 )
        plt.ylabel("Residual Weighting")
        plt.xlabel("PC rank")
        image_name = " Difference between scores and DL reconstructed scores."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
        #note that most of discrepancy likely de to the algoithm calculating 
        #scores from residual but this check using the mean centered data, 
        #which would be related to the known renormalisation issues of NIPALS

        plt.plot( (pcaMC.spectral_loading[:12,:].T-
                     (np.inner(pcaMC.icomponent_weight[:12,:],
                               (data.T-pcaMC.centring).T,
                               )      
                     ).T)
                 )
        plt.ylabel("Residual Score")
        plt.xlabel("Raman Shift /cm$^{-1}$")
        image_name = " Difference between loadings and SiD reconstructed loadings."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
### ***   END  Data Calculations   ***


### ***   START Equations   ***
### Matrix Equations
# Generate FIGURE for the main PCA equation in its possible arrangements
# Row data matrix (spectra displayed as rows by convention)
        pcaMC.figure_DSLT("DSLT")
        pcaMC.figure_DSLT("SDL")
        pcaMC.figure_DSLT("LTSiD")
### Vector Equations
        component_index = 1
        pcaMC.figure_sldi(component_index)
        pcaMC.figure_lsdi(component_index)
### Algorithm Equations
# Function to generate sum of squares figure showing how PCA is initialised
        pcaMC.figure_DTD()
        pcaMC.figure_DTDw()
### ***   END Equations   ***

### ***   START Common Signal   ***
# scaling factor calculated for subtracting the common signal from the positive
# and negative constituents of a PC. Use non-mean centred data for clarity
        nPC = 13
        pcaNonMC = npls.nipals(
            X_data = data,
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "NA",
            pixel_axis = wavelength_axis,
            spectral_weights = molar_profile,
            min_spectral_values = min_data,
        )
        pcaNonMC.calc_PCA()
# generate subtraction figures for positive vs negative score weighted sums.
        pcaNonMC.calc_Constituents(nPC)
        xview = range(625, 685)  # zoom in on region to check in detail for changes
        component_index = 2 #PC1 is not interesting for common signal (there is none, so have component_index>=2)
        pcaNonMC.figure_lpniScoreEqn(component_index)
# generate overlays of the constituents at adjusted scaling factors
        pcaNonMC.figure_lpniCommonSignalScalingFactors(nPC, xview)
        pcaNonMC.figure_lpniCommonSignal(component_index)
        pcaNonMC.figure_lpniCommonSignal(component_index, pcaNonMC.optSF[component_index-1]*1.01)#SF 1% larger
        pcaNonMC.figure_lpniCommonSignal(component_index, pcaNonMC.optSF[component_index-1]*0.99)#Sf 1% smaller
        
### ***   END  Common Signal   ***

### ***   END  Scree Plot   ***
        #create a molarprofile with no correlation
        FAmeans = np.mean(molar_profile,1)
        FAsd = np.std(molar_profile,1)
        #generate random numbers in array same size as molar_profile, scale by 1FAsd then add on mean value
        molar_profileUncorrRand = np.random.randn(*molar_profile.shape)
        molar_profile_Uncorr = ((molar_profileUncorrRand.T*np.mean(FAsd))+np.mean(FAmeans)).T #no covariance imposed by means and SD of FA
        molar_profile_Uncorr = 100* molar_profile_Uncorr/np.sum(molar_profile_Uncorr,0) # restandardise total to 100
        molar_profile_Variation_Uncorr = ((molar_profileUncorrRand.T*FAsd)+FAmeans).T # this has covariance imposed by means of FA


        #create a molar_profile with only 2 PCs
        nComp = 2
        pca = PCA(nComp)
        pca.fit(molar_profile)
        molar_profile2PC = np.dot(pca.transform(molar_profile)[:,:nComp],pca.components_[:nComp,:])
        molar_profile2PC += pca.mean_

            #Now generate simulated spectra for each sample by multiplying the simualted FA reference spectra by the FA profiles
        data_2covariance = np.dot(simplified_fatty_acid_spectra['simFA'],molar_profile2PC)
        data_0covariance = np.dot(simplified_fatty_acid_spectra['simFA'],molar_profile_Uncorr)
        data_0_Var_covariance = np.dot(simplified_fatty_acid_spectra['simFA'],molar_profile_Variation_Uncorr)
        pca_2_Cov = npls.nipals(
            X_data=data_2covariance,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_2_Cov.calc_PCA()
        pca_0_Cov = npls.nipals(
            X_data=data_0covariance,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_0_Cov.calc_PCA()
        pca_0_Var_Cov = npls.nipals(
            X_data=data_0_Var_covariance,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_0_Var_Cov.calc_PCA()
        
        shot_noise = np.random.randn(np.shape(data)[0],np.shape(data)[1])/10
        data_q_noise = data + ((data**0.5 + 10) * shot_noise / 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        data_1_noise = data + ((data**0.5 + 10) * shot_noise) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        data_4_noise = data + ((data**0.5 + 10) * shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        pca_q_noise = npls.nipals(
            X_data=data_q_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_q_noise.calc_PCA()
        pca_1_noise = npls.nipals(
            X_data=data_1_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_1_noise.calc_PCA()
        pca_4_noise = npls.nipals(
            X_data=data_4_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_4_noise.calc_PCA()

        corrPCs_noiseq = np.inner(pca_q_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise1 = np.inner(pca_1_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise4 = np.inner(pca_4_noise.spectral_loading,pcaMC.spectral_loading)
        #loadings already standardised to unit norm
        maxCorr_noiseq = np.max(np.abs(corrPCs_noiseq),axis=0)
        maxCorr_noise1 = np.max(np.abs(corrPCs_noise1),axis=0)
        maxCorr_noise4 = np.max(np.abs(corrPCs_noise4),axis=0)


        figScree, axScree = plt.subplots(2, 6, figsize=[16,10])
        figScree.subplots_adjust(wspace = 0.35)
        axScree[0,0] = plt.subplot2grid((2,6),(0,0),colspan=2, rowspan=1)
        axScree[0,1] = plt.subplot2grid((2,6),(0,2),colspan=2, rowspan=1)
        axScree[0,2] = plt.subplot2grid((2,6),(0,4),colspan=2, rowspan=1)
        axScree[1,0] = plt.subplot2grid((2,6),(1,0),colspan=3, rowspan=1)
        axScree[1,3] = plt.subplot2grid((2,6),(1,3),colspan=3, rowspan=1)
#        axScree[0,0].plot(range(10), pcaMC.Eigenvalue[:10]**0.5, "k")
#        axScree[0,0].plot(range(10), pca_2_Cov.Eigenvalue[:10]**0.5, "b")
#        axScree[0,0].plot(range(10), pca_0_Cov.Eigenvalue[:10]**0.5, "r")
#        axScree[0,0].plot(range(10), pca_0_Var_Cov.Eigenvalue[:10]**0.5, "g")
#        axScree[0,0].set_ylabel("Eigenvalue")
#        axScree[0,0].set_xlabel("PC rank")
        
#        tot_variance_data = sum(pcaMC.Eigenvalue**0.5)
#        axScree[0,1].plot(range(10), np.cumsum(100*(pcaMC.Eigenvalue[:10]**0.5)/tot_variance_data), "k")
#        tot_variance_2_Cov = sum(pca_2_Cov.Eigenvalue**0.5)
#        axScree[0,1].plot(range(10), np.cumsum(100*(pca_2_Cov.Eigenvalue[:10]**0.5)/tot_variance_2_Cov), "b")
#        tot_variance_0_Cov = sum(pca_0_Cov.Eigenvalue**0.5)
#        axScree[0,1].plot(range(10), np.cumsum(100*(pca_0_Cov.Eigenvalue[:10]**0.5)/tot_variance_0_Cov), "r")
#        tot_variance_0_Var_Cov = sum(pca_0_Var_Cov.Eigenvalue**0.5)
#        axScree[0,1].plot(range(10), np.cumsum(100*(pca_0_Var_Cov.Eigenvalue[:10]**0.5)/tot_variance_0_Var_Cov), "g")
#        axScree[0,1].set_ylabel("% Variance explained")
#        axScree[0,1].set_xlabel("PC rank")
            
        
        axScree[0,0].plot(range(1,11), pcaMC.Eigenvalue[:10]**0.5, "k")
        axScree[0,0].plot(range(1,11), pca_q_noise.Eigenvalue[:10]**0.5, "b")
        axScree[0,0].plot(range(1,11), pca_1_noise.Eigenvalue[:10]**0.5, "r")
        axScree[0,0].plot(range(1,11), pca_4_noise.Eigenvalue[:10]**0.5, "g")
        axScree[0,0].set_ylabel("Eigenvalue")
        axScree[0,0].set_xlabel("PC rank")
        axScree[0,0].legend(("Noiseless","SD 0.25","SD 1","SD 4"))
        axScree[0,0].annotate(
            "a)",
            xy=(0.12,0.9),
            xytext=(0.133,0.86),
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )

        axScree[0,1].plot(range(1,11), np.cumsum(100*(pcaMC.Eigenvalue[:10]**0.5)/sum(pcaMC.Eigenvalue**0.5)), "k")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_q_noise.Eigenvalue[:10]**0.5)/sum(pca_q_noise.Eigenvalue**0.5)), "b")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_1_noise.Eigenvalue[:10]**0.5)/sum(pca_1_noise.Eigenvalue**0.5)), "r")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_4_noise.Eigenvalue[:10]**0.5)/sum(pca_4_noise.Eigenvalue**0.5)), "g")
        axScree[0,1].set_ylabel("% Variance explained")
        axScree[0,1].set_xlabel("PC rank")
        axScree[0,1].annotate(
            "b)",
            xy=(0.25,0.9),
            xytext=(0.405,0.86),
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axScree[0,2].plot(range(1,11), np.diagonal(corrPCs_noise4)[:10], color = (0, 0.75, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise4[:10], color = (0.25, 1, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.diagonal(corrPCs_noise1)[:10], color = (0.75, 0, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise1[:10], color = ( 1 , 0.25, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.diagonal(corrPCs_noiseq)[:10], color = (0, 0, 0.75))
        axScree[0,2].plot(range(1,11), maxCorr_noiseq[:10], color = (0.25, 0.5, 1), linewidth=0.75)
        axScree[0,2].set_ylabel("Correlation vs noiseless")
        axScree[0,2].set_xlabel("PC rank")
        axScree[0,2].annotate(
            "c)",
            xy=(0.25,0.9),
            xytext=(0.67,0.86),
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )

        axScree[1,0].plot(pca_4_noise.pixel_axis, pca_4_noise.spectral_loading[:4].T-[0,0.3,0.6,0.9],color = (0, 1, 0),linewidth=1.75)
        axScree[1,0].plot(pca_1_noise.pixel_axis, pca_1_noise.spectral_loading[:4].T-[0,0.3,0.6,0.9],color = (1, 0.15, 0.15),linewidth=1.5)
        axScree[1,0].plot(pca_q_noise.pixel_axis, pca_q_noise.spectral_loading[:4].T-[0,0.3,0.6,0.9],color = (0.3, 0.3, 1),linewidth=1.25)
        axScree[1,0].plot(pcaMC.pixel_axis, pcaMC.spectral_loading[:4].T-[0,0.3,0.6,0.9],color = (0, 0, 0),linewidth = 1)
        axScree[1,0].set_ylabel("Weight")
        axScree[1,0].set_xticklabels([])
        axScree[1,0].set_yticklabels([])
        axScree[1,0].annotate(
            "d)",
            xy=(0.25,0.9),
            xytext=(0.133,0.44),
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axScree[1,3].plot(pca_4_noise.pixel_axis, pca_4_noise.spectral_loading[:4].T-[0,0.2,0.4,0.6]-pcaMC.spectral_loading[:4].T-[0,0.2,0.4,0.6], color = (0, 1, 0),linewidth=3)
        axScree[1,3].plot(pca_1_noise.pixel_axis, pca_1_noise.spectral_loading[:4].T-[0,0.2,0.4,0.6]-pcaMC.spectral_loading[:4].T-[0,0.2,0.4,0.6], color = (1, 0.15, 0.15), linewidth=2)
        axScree[1,3].plot(pca_q_noise.pixel_axis, pca_q_noise.spectral_loading[:4].T-[0,0.2,0.4,0.6]-pcaMC.spectral_loading[:4].T-[0,0.2,0.4,0.6], color = (0.3, 0.3, 1),linewidth=1)
        axScree[1,3].set_ylabel("Weight")
        axScree[1,3].set_yticklabels([])
        axScree[1,3].set_xticklabels([])
        axScree[1,3].annotate(
            "e)",
            xy=(0.25,0.9),
            xytext=(0.54,0.44),
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        image_name = " Comparing PCAs for different levels of noise"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
            
        plt.close()
### ***   END  Scree Plot   ***

### ******      END CLASS      ******
        return
