import numpy as np
import scipy.io as sio
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import patches
#from matplotlib import transforms
from sklearn.decomposition import PCA
#from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
import src.local_nipals as npls
from src.file_locations import data_folder,images_folder
# This expects to be called inside the jupyter project folder structure.


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
        mpl.rcParams['lines.markersize'] = 3

        GC_data = sio.loadmat(
            data_folder / "FA profile data.jrb", struct_as_record=False
        )
# Gas Chromatograph (GC) data is from Beattie et al. Lipids 2004 Vol 39 (9):897-906
        simplified_fatty_acid_spectra = sio.loadmat(data_folder / "FA spectra.mat", struct_as_record=False)
# simplified_fatty_acid_spectra are simplified spectra of fatty acid methyl esters built from the properties described in
# Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        wavelength_axis = simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][201:,]
        min_spectral_values = np.tile(
            np.min(simplified_fatty_acid_spectra["simFA"][201:,:], axis=1), (np.shape(simplified_fatty_acid_spectra["simFA"][201:,:])[1], 1)
        )
# convert the mass base profile provided into a molar profile
        molar_profile = GC_data["ANNSBUTTERmass"] / simplified_fatty_acid_spectra["FAproperties"][0, 0].MolarMass
        molar_profile = 100.0 * molar_profile / sum(molar_profile)
        sam_codes = GC_data["ANNSBUTTERhead"] 
        lac_Stage = np.empty(np.size(sam_codes),dtype=bool)
        feed = np.empty(np.size(sam_codes))
        week = np.empty(np.size(sam_codes))
        for iSam in range(sam_codes.shape[1]):
            lac_Stage[iSam] = sam_codes[0,iSam][0][0]=='L'
            feed[iSam] = np.int(sam_codes[0,iSam][0][1])
            week[iSam] = np.int(sam_codes[0,iSam][0][2:4])
        FAprops_N_Carbon = GC_data["N_Carbon"]
        FAprops_N_Olefin = GC_data["N_Olefin"]
        FAprops_N_Isomer = GC_data["N_Isomer"]
        

### generate simulated observational
# spectra for each sample by multiplying the simulated FA reference spectra by 
# the Fatty Acid profiles. Note that the simplified_fatty_acid_spectra spectra 
# have a standard intensity in the carbonyl mode peak (the peak with the 
# highest pixel position)
        data = np.dot(simplified_fatty_acid_spectra["simFA"][201:,:], molar_profile)
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

### Plot Pure Simulated Data & Plot Simulated Observational Data
        figData, axData = plt.subplots(1, 2, figsize=pcaMC.fig_Size)
        axData[0] = plt.subplot2grid((1, 11), (0, 0), colspan=5)
        axData[1] = plt.subplot2grid((1, 11), (0, 6), colspan=5)
        
        axData[0].plot( wavelength_axis , simplified_fatty_acid_spectra["simFA"][201:,:] )
        axData[0].set_ylabel("Intensity / Counts")
        axData[0].set_xlabel("Raman Shift cm$^{-1}$")
        axData[0].annotate(
            "a)",
            xy=(0.2, 0.95),
            xytext=(0.15, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axData[0].annotate(
            r'$\delta$ H-C=$_c$',
            xy=(0.2, 0.04),
            xytext=(1260, 9.5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[1, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )
        axData[0].annotate(
            r'$\delta$ C-H$_2$',
            xy=(1305, 9),
            xytext=(1305, 11),
            textcoords="data",
            xycoords="data",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0.8, 0.65, 0.3],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )
        axData[0].annotate(
            r'$\delta$C-H$_x$',
            xy=(0.2, 0.04),
            xytext=(1440, 8.5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0.87, 0.72, 0.38],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )
        axData[0].annotate(
            r'$\nu$C=C$_c$',
            xy=(0.2, 0.04),
            xytext=(1640, 12.5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[1, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )        
        axData[0].annotate(
            r'$\nu$C=C$_t$',
            xy=(0.2, 0.04),
            xytext=(1675, 9.5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 1, 0],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )        
        axData[0].annotate(
            r'$\nu$C=O',
            xy=(0.2, 0.04),
            xytext=(1745, 2),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 1],
            horizontalalignment="center",
            rotation=90,
            va="center",
        )        

        axData[1].plot( wavelength_axis , data  )
        axData[1].set_ylabel("Intensity / Counts")
        axData[1].set_xlabel("Raman Shift cm$^{-1}$")
        axData[1].annotate(
            "b)",
            xy=(0.2, 0.95),
            xytext=(0.57, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )

        image_name = " Simulated Spectra."
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
        image_name = " Local NIPALS minus SKLearn SVD vs Rank"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.'+ pcaMC.fig_Format)
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

        plt.plot(pcaMC.spectral_loading[0,:], linewidth=2.5)
        plt.plot(pcaMC.spectral_loading[18,:], linewidth=0.75)
        plt.plot(pcaMC.spectral_loading[0,:]-pcaMC.spectral_loading[18,:])
        plt.ylabel("Weighting")
        plt.xlabel("Raman Shift /cm$^{-1}$")
        image_name = " Difference between PC1 and 19"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
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

### Reconstructing signal from PCA
        figsldiRecon, axsldiRecon = plt.subplots(2, 5, figsize=pcaMC.fig_Size)
        axsldiRecon[0,0] = plt.subplot2grid((2, 26), (0, 0), colspan=8)
        axsldiRecon[0,1] = plt.subplot2grid((2, 26), (0, 8), colspan=1, rowspan=2)
        axsldiRecon[0,2] = plt.subplot2grid((2, 26), (0, 9), colspan=8)
        axsldiRecon[0,3] = plt.subplot2grid((2, 26), (0, 17), colspan=1, rowspan=2)
        axsldiRecon[0,4] = plt.subplot2grid((2, 26), (0, 18), colspan=8)
        axsldiRecon[1,0] = plt.subplot2grid((2, 26), (1, 0), colspan=8)
        axsldiRecon[1,2] = plt.subplot2grid((2, 26), (1, 9), colspan=8)
        axsldiRecon[1,4] = plt.subplot2grid((2, 26), (1, 18), colspan=8)

        pltDat = np.zeros([pcaMC.N_Vars,3])
        iSamZer = np.argmin(
            np.sum(np.abs(pcaMC.component_weight[:, 0:2]),1)
        )  # sample with the lowest total score across 1st 3 PCs
        pltDat[:,0] = pcaMC.data[:, iSamZer]
        iSamMin = np.argmin(pcaMC.component_weight[:, 1])
        pltDat[:,1] = pcaMC.data[:, iSamMin]
        iSamMax = np.argmax(pcaMC.component_weight[:, 2])
        pltDat[:,2] = pcaMC.data[:, iSamMax]
        pltDatMn = np.mean(np.abs(pltDat))*5
        axsldiRecon[0,0].plot(pltDat+pltDatMn*np.arange(3),LineWidth=2)
        axsldiRecon[0,2].plot(pltDat+pltDatMn*np.arange(3),LineWidth=2)
        axsldiRecon[0,4].plot(pltDat+pltDatMn*np.arange(3),LineWidth=2)
        reconDat =  np.zeros([pcaMC.N_Vars,3])
        reconDati =  np.zeros([pcaMC.N_Vars,3])
        for pc in range(3):
            reconDat[:,0] = np.inner(
                pcaMC.component_weight[iSamZer, :pc+1],
                pcaMC.spectral_loading[:pc+1, :].T,
            )
            reconDat[:,1] = np.inner(
                pcaMC.component_weight[iSamMin, :pc+1],
                pcaMC.spectral_loading[:pc+1, :].T,
            )
            reconDat[:,2] = np.inner(
                pcaMC.component_weight[iSamMax, :pc+1],
                pcaMC.spectral_loading[:pc+1, :].T,
            )
            reconDati[:,0] = np.inner(
                pcaMC.component_weight[iSamZer, pc+1],
                pcaMC.spectral_loading[pc+1, :].T,
            )
            reconDati[:,1] = np.inner(
                pcaMC.component_weight[iSamMin, pc+1],
                pcaMC.spectral_loading[pc+1, :].T,
            )
            reconDati[:,2] = np.inner(
                pcaMC.component_weight[iSamMax, pc+1],
                pcaMC.spectral_loading[pc+1, :].T,
            )
            axsldiRecon[0,pc*2].plot(reconDat+pltDatMn*np.arange(3),'--k',LineWidth=1)
            axsldiRecon[1,pc*2].plot(pltDat-reconDat+pltDatMn/2*np.arange(3))   
            axsldiRecon[1,pc*2].plot(reconDati+pltDatMn/2*np.arange(3),'--k')   

        axsldiRecon[1,2].set_ylim( axsldiRecon[1,0].get_ylim())
        axsldiRecon[1,4].set_ylim( axsldiRecon[1,0].get_ylim())
        axsldiRecon[0,0].legend(("o="+str(iSamZer), "o="+str(iSamMin), 
                                 "o="+str(iSamMax), "Recon"), 
                                fontsize='xx-small', 
                                loc=9)
        axsldiRecon[0,0].annotate(
            "a) PC1 reconstruction",
            xy=(0.22, 0.95),
            xytext=(0.22, 0.9),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[0,2].annotate(
            "b) PC1:2 reconstruction",
            xy=(0.5, 0.95),
            xytext=(0.5, 0.9),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[0,4].annotate(
            "c) PC1:3 reconstruction",
            xy=(0.84, 0.9),
            xytext=(0.77, 0.9),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
            
        axsldiRecon[1,0].annotate(
            "d) PC2 reconstruction",
            xy=(0.22, 0.95),
            xytext=(0.22, 0.45),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[1,2].annotate(
            "e) PC3 reconstruction",
            xy=(0.5, 0.55),
            xytext=(0.5, 0.45),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[1,4].annotate(
            "f) PC4 reconstruction",
            xy=(0.82, 0.95),
            xytext=(0.77, 0.45),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )

        axsldiRecon[0,1].annotate(
            "",
            xy=(0.4, 0.7),
            xytext=(0.35, 0.7),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRecon[0,1].annotate(
            r"$l_{1:2} \cdot s_{o,1:2}$",
            xy=(0.375, 0.71),
            xytext=(0.375, 0.72),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )

        axsldiRecon[0,3].annotate(
            "",
            xy=(0.67, 0.7),
            xytext=(0.62, 0.7),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRecon[0,3].annotate(
            r"$l_{1:3} \cdot s_{o,1:3}$",
            xy=(0.645, 0.71),
            xytext=(0.645, 0.72),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[0,1].annotate(
            "",
            xy=(0.4, 0.3),
            xytext=(0.35, 0.3),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRecon[0,1].annotate(
            r"$d_o - l_{2} \cdot s_{o,2}$",
            xy=(0.375, 0.31),
            xytext=(0.375,      0.32),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRecon[0,3].annotate(
            "",
            xy=(0.67, 0.3),
            xytext=(0.62, 0.3),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRecon[0,3].annotate(
            r"$d_o - l_{3} \cdot s_{o,3}$",
            xy=(0.645, 0.31),
            xytext=(0.645, 0.32),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
    
        if not pcaMC.fig_Show_Values: 
            for iax in range(len(axsldiRecon)):
                for iax2 in range(5):
                    axsldiRecon[iax,iax2].axis("off")
            
        if pcaMC.fig_Show_Labels:
            for iax in range(len(axsldiRecon)):
                for iax2 in range(3):
                    axsldiRecon[iax,iax2*2].set_ylabel(pcaMC.fig_Y_Label)
                    axsldiRecon[iax,iax2*2].set_xlabel(pcaMC.fig_X_Label)
            
        image_name = " Reconstructed data."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figsldiRecon.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)

#        plt.show()
        plt.close()


### ***   START Common Signal   ***
# scaling factor calculated for subtracting the common signal from the positive
# and negative constituents of a PC. Use non-mean centerd data for clarity
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
        xview = range(425, 485)  # zoom in on region to check in detail for changes
        component_index = 2 #PC1 is not interesting for common signal (there is none, so have component_index>=2)
        pcaNonMC.figure_lpniScoreEqn(component_index)
# generate overlays of the constituents at adjusted scaling factors
        pcaNonMC.figure_lpniCommonSignalScalingFactors(nPC, xview)
        pcaNonMC.figure_lpniCommonSignal(component_index)
        pcaNonMC.figure_lpniCommonSignal(component_index, pcaNonMC.optSF[component_index-1]*1.01)#SF 1% larger
        pcaNonMC.figure_lpniCommonSignal(component_index, pcaNonMC.optSF[component_index-1]*0.99)#Sf 1% smaller
        
### ***   END  Common Signal   ***

### ***   Start  Interpretation   ***
### Generate Noisy Data
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
        data_2covariance = np.dot(simplified_fatty_acid_spectra['simFA'][201:,:],molar_profile2PC)
        data_0covariance = np.dot(simplified_fatty_acid_spectra['simFA'][201:,:],molar_profile_Uncorr)
        data_0_Var_covariance = np.dot(simplified_fatty_acid_spectra['simFA'][201:,:],molar_profile_Variation_Uncorr)
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
        majpk = np.where(np.mean(data,axis=1)>np.mean(data))        
        signal = np.mean(data[majpk,:],axis=1)
        data_q_noise = data + ((data**0.5 + 10) * shot_noise / 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in quarter scaled noise: ' + 
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10) 
                                        * shot_noise[majpk,:] / 4,axis=1))))
        data_1_noise = data + ((data**0.5 + 10) * shot_noise) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in unscaled noise: ' + 
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10) 
                                        * shot_noise[majpk,:],axis=1))))
        data_4_noise = data + ((data**0.5 + 10) * shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in 4 times scaled noise: ' + 
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10) 
                                        * shot_noise[majpk,:]*4,axis=1))))
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

        pca_noise = npls.nipals(
            X_data=((data**0.5 + 10) * shot_noise),
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_noise.calc_PCA()# noise from SNR 100

        corrPCs_noiseq = np.inner(pca_q_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise1 = np.inner(pca_1_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise4 = np.inner(pca_4_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise = np.inner(pca_noise.spectral_loading,pcaMC.spectral_loading)
        #loadings already standardised to unit norm
        corrPCs_noiseq_R2sum_ax0 = np.sum(corrPCs_noiseq**2,axis=0)#total variance shared between each noiseq loading and the noisless loadings
        corrPCs_noiseq_R2sum_ax1 = np.sum(corrPCs_noiseq**2,axis=1)#total variance shared between each noiseless PC and the noiseq loadings

        maxCorr_noiseq = np.max(np.abs(corrPCs_noiseq),axis=0)
        maxCorr_noise1 = np.max(np.abs(corrPCs_noise1),axis=0)
        maxCorr_noise4 = np.max(np.abs(corrPCs_noise4),axis=0)
        maxCorr_noise = np.max(np.abs(corrPCs_noise),axis=0)
        maxCorrMean = ((np.sum(maxCorr_noise**2)**0.5)/
                maxCorr_noise.shape[0]**0.5) #correlation measures noise so propagate as variance
        print('Mean optimal Correlation : ' + str(maxCorrMean))
        print('SE Correlation : ' + str(maxCorrMean + [-np.std(maxCorr_noise),np.std(maxCorr_noise)]))
        
        max_Ixq = np.empty(*maxCorr_noiseq.shape)
        max_Ix1 = np.copy(max_Ixq)
        max_Ix4 = np.copy(max_Ixq)
        max_IxN = np.copy(max_Ixq)
        max_Snq = np.copy(max_Ixq)
        max_Sn1 = np.copy(max_Ixq)
        max_Sn4 = np.copy(max_Ixq)
        max_SnN = np.copy(max_Ixq)
        
        for iCol in range(np.shape(maxCorr_noiseq)[0]):
            max_Ixq[iCol] = np.where(np.abs(corrPCs_noiseq[:,iCol])==maxCorr_noiseq[iCol])[0]
            max_Snq[iCol] = np.sign(corrPCs_noiseq[max_Ixq[iCol].astype(int),iCol])
            max_Ix1[iCol] = np.where(np.abs(corrPCs_noise1[:,iCol])==maxCorr_noise1[iCol])[0]
            max_Sn1[iCol] = np.sign(corrPCs_noise1[max_Ix1[iCol].astype(int),iCol])
            max_Ix4[iCol] = np.where(np.abs(corrPCs_noise4[:,iCol])==maxCorr_noise4[iCol])[0]
            max_Sn4[iCol] = np.sign(corrPCs_noise4[max_Ix4[iCol].astype(int),iCol])
            max_IxN[iCol] = np.where(np.abs(corrPCs_noise[:,iCol])==maxCorr_noise[iCol])[0]
            max_SnN[iCol] = np.sign(corrPCs_noise[max_IxN[iCol].astype(int),iCol])

### Scree Plot
        figScree, axScree = plt.subplots(2, 3, figsize=[16,10])
        figScree.subplots_adjust(wspace = 0.35)
        axScree[0,0] = plt.subplot2grid((2,6),(0,0),colspan=2, rowspan=1)
        axScree[0,1] = plt.subplot2grid((2,6),(0,2),colspan=2, rowspan=1)
        axScree[0,2] = plt.subplot2grid((2,6),(0,4),colspan=2, rowspan=1)
        axScree[1,0] = plt.subplot2grid((2,6),(1,0),colspan=3, rowspan=1)
        axScree[1,2] = plt.subplot2grid((2,6),(1,3),colspan=3, rowspan=1)
                    
        axScree[0,0].plot(range(1,11), pcaMC.Eigenvalue[:10]**0.5, "k")
        axScree[0,0].plot(range(1,11), pca_q_noise.Eigenvalue[:10]**0.5, "b")
        axScree[0,0].plot(range(1,11), pca_1_noise.Eigenvalue[:10]**0.5, "r")
        axScree[0,0].plot(range(1,11), pca_4_noise.Eigenvalue[:10]**0.5, "g")
        axScree[0,0].plot(range(1,11), 0.25*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 0.25*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 0.5, 1])
        axScree[0,0].plot(range(1,11), pca_noise.Eigenvalue[:10]**0.5, "--",color=[1, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 4*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 1, 0.5])
        axScree[0,0].set_ylabel("Eigenvalue")
        axScree[0,0].set_xlabel("PC rank")
        axScree[0,0].legend(("$SNR_\infty$","$SNR_{400}$","$SNR_{100}$","$SNR_{25}$","Noise"))

        axScree[0,1].plot(range(1,11), np.cumsum(100*(pcaMC.Eigenvalue[:10]**0.5)/sum(pcaMC.Eigenvalue**0.5)), "k")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_q_noise.Eigenvalue[:10]**0.5)/sum(pca_q_noise.Eigenvalue**0.5)), "b")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_1_noise.Eigenvalue[:10]**0.5)/sum(pca_1_noise.Eigenvalue**0.5)), "r")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_4_noise.Eigenvalue[:10]**0.5)/sum(pca_4_noise.Eigenvalue**0.5)), "g")
#        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_noise.Eigenvalue[:10]**0.5)/sum(pca_4_noise.Eigenvalue**0.5)), "--",color=[0.45,0.45,0.45])
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_noise.Eigenvalue[:10]**0.5)/sum(pca_noise.Eigenvalue**0.5)), "--",color=[0.75,0.75,0.75])
        axScree[0,1].set_ylabel("% Variance explained")
        axScree[0,1].set_xlabel("PC rank")

        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise4)[:10]), color = (0, 0.75, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise4[:10], color = (0.25, 1, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise1)[:10]), color = (0.75, 0, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise1[:10], color = ( 1 , 0.25, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noiseq)[:10]), color = (0, 0, 0.75))
        axScree[0,2].plot(range(1,11), maxCorr_noiseq[:10], color = (0.25, 0.5, 1), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise)[:10]), color = (0.45, 0.45, 0.45))
        axScree[0,2].plot(range(1,11), maxCorr_noise[:10], color = (0.75, 0.75, 0.75), linewidth=0.75)
        axScree[0,2].set_ylabel("Correlation vs noiseless")
        axScree[0,2].set_xlabel("PC rank")

        axScree[1,0].plot(pca_noise.pixel_axis, pca_noise.spectral_loading[:5].T-[0,0.3,0.6,0.9,1.2],color = (0.75, 0.75, 0.75),linewidth=2)
        axScree[1,0].plot(pca_4_noise.pixel_axis, pca_4_noise.spectral_loading[:5].T-[0,0.3,0.6,0.9,1.2],color = (0, 1, 0),linewidth=1.75)
        axScree[1,0].plot(pca_1_noise.pixel_axis, pca_1_noise.spectral_loading[:5].T-[0,0.3,0.6,0.9,1.2],color = (1, 0.15, 0.15),linewidth=1.5)
        axScree[1,0].plot(pca_q_noise.pixel_axis, pca_q_noise.spectral_loading[:5].T-[0,0.3,0.6,0.9,1.2],color = (0.3, 0.3, 1),linewidth=1.25)
        axScree[1,0].plot(pcaMC.pixel_axis, pcaMC.spectral_loading[:5].T-[0,0.3,0.6,0.9,1.2],color = (0, 0, 0),linewidth = 1)
        axScree[1,0].set_ylabel("Weight")
        axScree[1,0].set_xticklabels([])
        axScree[1,0].set_yticklabels([])
        # 1 create 3 noise variant replicates of one sample and plot overlaid
        # 2 reconstruct replicates from noiseless PCA and plot overlaid, offset from 1
        # 3 subtract noiseless original spectrum then plot residuals overlaid, offset from 2. Scale up if necessary to compare, annotate with scaling used
        reps_4_noise = np.tile(data[:,23],(data.shape[1],1)).T
        reps_4_noise = reps_4_noise + ((reps_4_noise**0.5 + 10) * shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        reps_4_noise_recon = pcaMC.reduced_Rank_Reconstruction( reps_4_noise , 10 )
        
        axScree[1,2].plot(pca_noise.pixel_axis, reps_4_noise)
        offset = np.mean(reps_4_noise[:])
        axScree[1,2].plot(pca_noise.pixel_axis, reps_4_noise_recon-offset)
        axScree[1,2].plot(pca_noise.pixel_axis, (reps_4_noise.T-data[:,23]).T-1.5*offset)
        axScree[1,2].plot(pca_noise.pixel_axis, (reps_4_noise_recon.T-data[:,23]).T-2.5*offset)
        axScree[1,2].plot(pca_noise.pixel_axis, 10*(reps_4_noise_recon.T-data[:,23]).T-4.5*offset)
        axScree[1,2].set_ylabel("Intensity")
        axScree[1,2].set_yticklabels([])
        axScree[1,2].set_xticklabels([])

        subLabels = [r"a) Scree Plot", r"b) Cummulative Variance Explained",
                     r"c) Correlation vs Noiseless", r"d) Loadings", r"skip", r"e) Reconstructed Data"]
        for ax1 in range(2):
            for ax2 in range(3):
                if (ax1==1 and ax2==1)==False:
                    axScree[ax1,ax2].annotate(
                        subLabels[ax1*3+ax2],
                        xy=(0.18, 0.89),
                        xytext=(0.1, 1.05),
                        textcoords="axes fraction",
                        xycoords="axes fraction",
                        horizontalalignment="left",
                    )

        subsubStr = ["i", "ii", "iii", "iv","v"]
        ypos = [0, -offset, -1.5*offset, -2.5*offset, -4.5*offset]
        for subsub in np.arange(5):
            axScree[1,2].annotate(
                subsubStr[subsub]+")",
                xy=(pca_noise.pixel_axis[0]*0.98,ypos[subsub]),
                xytext=(pca_noise.pixel_axis[0]*0.98,ypos[subsub]),
                xycoords="data",
                textcoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="left",
            )
            
        image_name = " Comparing PCAs for different levels of noise"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
            
        plt.close()
    
### Model Correlation Plot
        figNoiseCorr, axNoiseCorr = plt.subplots(1, 3, figsize=pcaMC.fig_Size)
        axNoiseCorr[0] = plt.subplot2grid((1,11),(0,0),colspan=3)
        axNoiseCorr[1] = plt.subplot2grid((1,11),(0,4),colspan=3)
        axNoiseCorr[2] = plt.subplot2grid((1,11),(0,8),colspan=3)

        axNoiseCorr[0].plot(range(1,81), corrPCs_noiseq[:,range(0,7)]**2,)
        axNoiseCorr[1].plot(range(1,81), corrPCs_noiseq[range(0,7),:].T**2,)
        axNoiseCorr[2].plot(range(1,81),corrPCs_noiseq_R2sum_ax1,linewidth=2)
        axNoiseCorr[2].plot(range(1,81),corrPCs_noiseq_R2sum_ax0,linewidth=1.5)
        axNoiseCorr[2].plot(range(1,81),maxCorr_noiseq**2,linewidth=1)
        axNoiseCorr[0].set_ylabel("R$^2$")
        axNoiseCorr[0].set_xlabel("PC rank SNR$_{400}$")
        axNoiseCorr[1].set_ylabel("R$^2$")
        axNoiseCorr[1].set_xlabel("PC rank SNR$_\infty$")
        axNoiseCorr[2].set_ylabel("Total R$^2$")
        axNoiseCorr[2].set_xlabel("PC rank")
        axNoiseCorr[0].legend(range(1,7),fontsize="small")
        axNoiseCorr[2].legend(("$\Sigma R^2_{PC400 \mapsto PC\infty}$",
                               "$\Sigma R^2_{PC\infty \mapsto PC400}$",
                               "$\max (R^2_{PC400 \mapsto PC\infty})$"),
                              fontsize="small")


        subLabels = [r"a) PC$_{SNR\infty} \mapsto PC_{SNR400}$", r"b) PC$_{SNR400} \mapsto PC_{SNR\infty}$",
                     r"c) $\Sigma  \, & \, \max \, R^2$"]
        for ax1 in range(3):
            axNoiseCorr[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0, 1.05),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )
        image_name = " PC correlations noisy vs noiseless "
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
        plt.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
            
        plt.close()
        
### Biplot

        figbiplots, axbiplots = plt.subplots(2, 3, figsize=[9,5.3])
        axbiplots[0,0] = plt.subplot2grid((14,22), (0, 0), colspan=6, rowspan=6)
        axbiplots[0,1] = plt.subplot2grid((14,22), (0, 8), colspan=6, rowspan=6)
        axbiplots[0,2] = plt.subplot2grid((14,22), (0, 16), colspan=6, rowspan=6)
        axbiplots[1,0] = plt.subplot2grid((14,22), (8, 0), colspan=6, rowspan=6)
        axbiplots[1,1] = plt.subplot2grid((14,22), (8, 8), colspan=6, rowspan=6)
        axbiplots[1,2] = plt.subplot2grid((14,22), (8, 16), colspan=6, rowspan=6)
        figbiplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        grps = [[0,0,2,2,4,4,6,6],[False,True,False,True,False,True,False,True],['db','db','or','or','^c','^c','sm','sm']]
        for iS in range(8):
            if grps[1][iS]:
                cGrp = lac_Stage
                fillS = 'full'
            else:
                cGrp = ~lac_Stage
                fillS = 'none'
            ix = np.where(np.logical_and(feed==grps[0][iS] , cGrp))[0]
            axbiplots[0,0].plot(pcaMC.component_weight[ix, 0],pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS)
            axbiplots[0,1].plot(pcaMC.component_weight[ix, 0],pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS)
            axbiplots[0,2].plot(week[ix],pcaMC.component_weight[ix, 0],grps[2][iS],fillstyle=fillS)
        
        selvars = [90,95,104,108,232,241,451,472,538,549]#np.argsort(np.sum(np.abs(pcaMC.spectral_loading[:2,:]),0))[-20:]
        lOff = np.round(-(np.max(pcaMC.spectral_loading[1,:])-np.min(pcaMC.spectral_loading[0,:]))*12.5)/10 #offset by a simple number
        axbiplots[1,0].plot(wavelength_axis[selvars],pcaMC.spectral_loading[0,selvars],'.r')
        axbiplots[1,0].plot(wavelength_axis[selvars],pcaMC.spectral_loading[1,selvars] + lOff,'.',color=[1, 0.3, 0.3])
        axbiplots[1,0].plot(wavelength_axis,pcaMC.spectral_loading[0,:],'k') 
        axbiplots[1,0].plot(wavelength_axis[[0,-1]],[0,0],'--',color=[0.6,0.6,0.6]) 
        axbiplots[1,0].plot(wavelength_axis,pcaMC.spectral_loading[1,:] + lOff,color=[0.3,0.3,0.3])
        axbiplots[1,0].plot(wavelength_axis[[0,-1]],[lOff,lOff],'--',color=[0.6,0.6,0.6]) 
        
        offset = 5 #spacing between line and annotation
        for iV in range(len(selvars)):
            p1 = pcaMC.spectral_loading[0,selvars[iV]]*pcaMC.Eigenvalue[0]**0.5
            p2 = pcaMC.spectral_loading[1,selvars[iV]]*pcaMC.Eigenvalue[1]**0.5
            axbiplots[0,1].plot([0,p1],[0,p2],'--o',color=[0.4,0.4,0.4])
            axbiplots[0,1].annotate(str(wavelength_axis[selvars[iV]]),
                                    xy=(p1, p2),
                                    xytext=(p1+offset*np.sign(p1), p2+offset*np.sign(p2)),
                                    textcoords="data",xycoords="data",
                                    fontsize=pcaMC.fig_Text_Size*0.75,
                                    horizontalalignment="center",
                                    color=[0.7,0.7,0.7],
                                    )
        
        loadAutoCorr = np.outer(pcaMC.spectral_loading[0,np.arange(pcaMC.N_Vars-1,-1,-1)],
                                pcaMC.spectral_loading[1,np.arange(pcaMC.N_Vars-1,-1,-1)])  # have to reverse order in order to get image right in figure
        loadAutoCorr = loadAutoCorr[:,np.arange(pcaMC.N_Vars-1,-1,-1)]/np.max(np.abs(loadAutoCorr))
        imDim = np.round(loadAutoCorr.shape[0]*1.1).astype('int')
        loDim = (imDim - loadAutoCorr.shape[0]).astype('int')
        loadAutoCorrIm = np.zeros([imDim,imDim,3])
        loadAutoCorrt = np.copy(loadAutoCorr)
        loadAutoCorrt[np.where(loadAutoCorrt<0)] = 0        
        loadAutoCorrIm[:loadAutoCorr.shape[0],loDim:,0] = loadAutoCorrt        
        loadAutoCorrIm[:loadAutoCorr.shape[0],loDim:,1] = loadAutoCorrt        
        loadAutoCorrt = -np.copy(loadAutoCorr) 
        loadAutoCorrt[np.where(loadAutoCorrt<0)] = 0        
        #        loadAutoCorrIm[loDim:,loDim:,1] = loadAutoCorrt
        loadAutoCorrIm[:loadAutoCorr.shape[0],loDim:,1] = loadAutoCorrt + loadAutoCorrIm[:loadAutoCorr.shape[0],loDim:,1]
        loadAutoCorrIm[:loadAutoCorr.shape[0],loDim:,2] = loadAutoCorrt
        axrng = [pcaMC.pixel_axis[0]-(pcaMC.pixel_axis[1]-pcaMC.pixel_axis[0])*loDim,pcaMC.pixel_axis[-1]]
        axbiplots[1,1].imshow(loadAutoCorrIm**0.5, extent=np.concatenate([axrng,axrng]))
        axrngL = [pcaMC.pixel_axis[0],pcaMC.pixel_axis[-1]]
        axbiplots[1,1].plot(axrngL,axrngL,'--w',lw=0.5)
        axbiplots[1,1].plot(wavelength_axis,pcaMC.spectral_loading[0,:]*loDim/(np.max(pcaMC.spectral_loading[0,:]))+axrng[0]+loDim/2 ,color=[0.65,0.75,1])
        axbiplots[1,1].plot(pcaMC.spectral_loading[1,:]*loDim/(np.max(pcaMC.spectral_loading[1,:]))+axrng[0]+loDim/2 , wavelength_axis,color=[0.75,0.65,1])
        
        # now plot score weighted observations for positive and negative plus loading for PC1. Only partially add back the mean (1/7)
        axbiplots[1,2].plot(pcaMC.pixel_axis,np.sum(np.dot(pcaMC.component_weight[pcaMC.component_weight[:,0]>0,0][:,None],
                                                           pcaMC.spectral_loading[0,:][None,:]),axis=0)/44+pcaMC.centring/7) #[:,None] converts 1D vector to a 2D vector with size 1 along None axis
        axbiplots[1,2].plot(pcaMC.pixel_axis,np.sum(np.dot(pcaMC.component_weight[pcaMC.component_weight[:,0]<0,0][:,None],
                                                           pcaMC.spectral_loading[0,:][None,:]),axis=0)/44+pcaMC.centring/7)
        axbiplots[1,2].legend(['+ve','-ve'],fontsize=pcaMC.fig_Text_Size*0.65)
        
        cPC = -1
        for iPC in range(2):
            shd = 0.2*iPC+0.2
            cPC = cPC+1 # allow plotting of PCs other than just 1 and 2
            aoff = lOff*cPC
            ix = np.where(pcaMC.spectral_loading[iPC,:]==np.min(pcaMC.spectral_loading[iPC,:]))
            xyB = (pcaMC.component_weight[np.where(pcaMC.component_weight[:,iPC]==np.min(pcaMC.component_weight[:,iPC])),iPC][0][0],0)
            xyA = ( pcaMC.pixel_axis[ix[0][0]] , pcaMC.spectral_loading[iPC,ix][0][0]+aoff)
            con = ConnectionPatch( xyA = xyA,
                 xyB = (xyB[cPC],xyB[(cPC-1)**2]),coordsA = 'data', coordsB='data', 
                 axesA=axbiplots[1,0], axesB=axbiplots[0,0],arrowstyle='->',
                 ls='dashed',color=[shd*0.5,shd,shd*2],lw=0.5)
            axbiplots[1,0].add_artist(con)
            axbiplots[1,0].plot(xyA[0],xyA[1],'.',color=[shd*0.5,shd,shd*2])
            ix = np.where(pcaMC.spectral_loading[iPC,:]==np.max(pcaMC.spectral_loading[iPC,:]))
            xyB = (pcaMC.component_weight[np.where(pcaMC.component_weight[:,iPC]==np.max(pcaMC.component_weight[:,iPC])),iPC][0][0],0)
            xyA = ( pcaMC.pixel_axis[ix[0][0]] , pcaMC.spectral_loading[iPC,ix][0][0]+aoff)
            con = ConnectionPatch(xyA=xyA,
                 xyB = (xyB[cPC],xyB[(cPC-1)**2]),coordsA = 'data', coordsB='data', 
                 axesA=axbiplots[1,0], axesB=axbiplots[0,0],arrowstyle='->',
                 ls='dashed',color=[shd*2,shd,shd*0.5],lw=0.5)
            axbiplots[1,0].add_artist(con)
            axbiplots[1,0].plot(xyA[0],xyA[1],'.',color=[shd*2,shd,shd*0.5])
        
        subLabels = ["a) Score-Score plot","b) Biplot", "c) Score Trend",
                     "d) Loadings line plot","e) Outer Product", "f) PC1 Score weighted data"]
        for ax1 in range(2):
            for ax2 in range(3):
                axbiplots[ax1,ax2].annotate(
                    subLabels[ax1*3+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
        
        axbiplots[0,0].set_xlabel('t[1]',labelpad=-2)
        axbiplots[0,0].set_ylabel('t[2]',labelpad=-2)
        axbiplots[0,1].set_xlabel('t[1]',labelpad=-2)
        axbiplots[0,1].set_ylabel('t[2]',labelpad=-2)
        axbiplots[0,2].set_xlabel('time (weeks)',labelpad=-2)
        axbiplots[0,2].set_ylabel('t[1]',labelpad=-2)
        axbiplots[0,0].legend(['0mg G0','0mg G1','2mg','_','4mg','_','6mg'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axbiplots[1,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-2)
        axbiplots[1,0].set_ylabel('Coefficient',labelpad=-2)
        axbiplots[1,1].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-2)
        axbiplots[1,1].set_ylabel('Raman Shift (cm$^{-1}$)',labelpad=-2)
        axbiplots[1,2].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-2)
        axbiplots[1,2].set_ylabel('Intensity (Counts)',labelpad=-2)
            
        image_name = " Interpretation plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figbiplots.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        #        plt.show()
        plt.close()
        
### Dataset Strategies
        # show moire pattern in dataset selection
        imGrps = np.zeros([8,11,3])
        imNFold = np.zeros([8,11,3])
        imMonteCarlo = np.zeros([8,11,3])
        np.random.seed(375746)
        rix = np.random.randint(0,87,88) #random index
        monteCarlo = rix>22
 
 
        for i in range(8): #scale so never black
            cix = np.arange(i*11,i*11+11) #current index
            imGrps[i,:,0] = (week[cix]+3)/23
            imGrps[i,:,1] = (feed[cix]+1)/7
            imGrps[i,:,2] = (lac_Stage[cix]+0.5)/1.5
            if i >1:
                imNFold[i,:,0] = (week[cix]+3)/23
                imNFold[i,:,1] = (feed[cix]+1)/7
                imNFold[i,:,2] = (lac_Stage[cix]+0.5)/1.5
            else:
                imNFold[i,:,0] = 0
                imNFold[i,:,1] = 0
                imNFold[i,:,2] = 0
                
 
            imMonteCarlo[i,:,0] = (week[cix]+3)/23*monteCarlo[cix]
            imMonteCarlo[i,:,1] = (feed[cix]+1)/7*monteCarlo[cix]
            imMonteCarlo[i,:,2] = (lac_Stage[cix]+0.5)/1.5*monteCarlo[cix]


        setMeansNFold = np.zeros([4,6])
        setMeansMonteCarlo = np.zeros([4,6])
        imSets = np.zeros([6,8,3])
        for i in range(4): #measure membership of groups in 4 folds
            cix = np.ones([88]).astype('bool')
            cix[np.arange(i*22,i*22+22)] = False
            setMeansNFold[i,0] = np.mean(week[cix])
            setMeansNFold[i,1] = np.mean(week[~cix])
            setMeansNFold[i,2] = np.mean(feed[cix])
            setMeansNFold[i,3] = np.mean(feed[~cix])
            setMeansNFold[i,4] = np.mean(lac_Stage[cix])
            setMeansNFold[i,5] = np.mean(lac_Stage[~cix])

            cix2 = np.ones([88]).astype('bool')
            cix2[rix[cix]] = False
            setMeansMonteCarlo[i,0] = np.mean(week[cix2])
            setMeansMonteCarlo[i,1] = np.mean(week[~cix2])
            setMeansMonteCarlo[i,2] = np.mean(feed[cix2])
            setMeansMonteCarlo[i,3] = np.mean(feed[~cix2])
            setMeansMonteCarlo[i,4] = np.mean(lac_Stage[cix2])
            setMeansMonteCarlo[i,5] = np.mean(lac_Stage[~cix2])

            imSets[:2,i,0] = (setMeansNFold[i,:2]+3)/23
            imSets[2:4,i,1] = (setMeansNFold[i,2:4]+1)/7
            imSets[4:,i,2] = (setMeansNFold[i,4:]+0.5)/1.5
            imSets[:2,i+4,0] = (setMeansMonteCarlo[i,:2]+3)/23
            imSets[2:4,i+4,1] = (setMeansMonteCarlo[i,2:4]+1)/7
            imSets[4:,i+4,2] = (setMeansMonteCarlo[i,4:]+0.5)/1.5
 

        figdataset, axdataset = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axdataset[0,0] = plt.subplot2grid((2, 2), (0, 0))
        plt.imshow(imGrps)
        axdataset[0,1] = plt.subplot2grid((2, 2), (0, 1))
        plt.imshow(imNFold)
        axdataset[1,0] = plt.subplot2grid((2, 2), (1, 0))
        plt.imshow(imMonteCarlo)
        axdataset[1,1] = plt.subplot2grid((2, 2), (1, 1))
        plt.imshow(imSets)

        subF = np.column_stack((['a)','c)'],['b)','d)']))
        if not pcaMC.fig_Show_Values: 
            for iax in range(2):
                for iax2 in range(2):
                    axdataset[iax,iax2].axis("off")
                    axdataset[iax,iax2].annotate(subF[iax,iax2],
                            xy=(0.25, 0),
                            xytext=(-1.5, -0.6),
                            textcoords="data",
                            xycoords="data",
                            fontsize=pcaMC.fig_Text_Size,
                            horizontalalignment="center", va="bottom",
                            
                        )

        axdataset[0,0].annotate("Week",
                xy=(0.3, 0.95),
                xytext=(0.3, 0.97),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=0.5', lw=1.5)
                
            )
        
        axdataset[0,0].annotate("0",
                xy=(0.25, 0),
                xytext=(0.05, -0.5),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="bottom",
                
            )
        axdataset[0,0].annotate("20",
                xy=(0.25, 0),
                xytext=(9.95, -0.5),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="bottom",
                
            )

        axdataset[0,0].annotate("Dose",
                xy=(0.04, 0.7),
                xytext=(0.02, 0.7),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=pcaMC.fig_Text_Size, rotation=90,
                horizontalalignment="center", va="center",
                arrowprops=dict(arrowstyle='-[, widthB=5, lengthB=0.5', lw=1.5)
                
            )
        
        axdataset[0,0].annotate("0",
                xy=(-0.1, 0),
                xytext=(-3.5, 0),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="left", va="top",
                
            )
        axdataset[0,0].annotate("200",
                xy=(-0.1, 0),
                xytext=(-3.5, 2),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="top",
                
            )
        axdataset[0,0].annotate("400",
                xy=(-0.1, 0),
                xytext=(-3.5, 4),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="top",
                
            )
        axdataset[0,0].annotate("600",
                xy=(-0.1, 0),
                xytext=(-3.5, 6),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="top",
                
            )

        axdataset[0,0].annotate("Group",
                xy=(0.11, 0.83),
                xytext=(0.09, 0.83),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=pcaMC.fig_Text_Size, rotation=90,
                horizontalalignment="center", va="center",
                arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.5', lw=1.5)
                
            )
        
        axdataset[0,0].annotate("E",
                xy=(-0.1, 0),
                xytext=(-1, 0),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                
            )
        axdataset[0,0].annotate("L",
                xy=(-0.1, 1.5),
                xytext=(-1, 1),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                
            )

        axdataset[1,1].annotate("K-Fold",
                xy=(0.65, 0.485),
                xytext=(0.65, 0.51),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict(arrowstyle='-[, widthB=3, lengthB=0.5', lw=1.5)
                
            )
        axdataset[1,1].annotate("Monte-Carlo",
                xy=(0.8, 0.485),
                xytext=(0.8, 0.51),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict(arrowstyle='-[, widthB=3, lengthB=0.5', lw=1.5)
                
            )

        subF = ['Week','Dose','Group']
        for i in range(3):
            axdataset[1,1].annotate("Train\nTest",
                    xy=(-0.1, 0),
                    xytext=(-1.5, i*2 + 0.5),
                    textcoords="data",
                    xycoords="data",
                    fontsize=pcaMC.fig_Text_Size, linespacing=1.75,
                    horizontalalignment="center", va="center",
                    
                )
            axdataset[1,1].annotate("Mean\n"+subF[i],
                    xy=(-0.1, 1.5),
                    xytext=(8.5, i*2 + 0.5),
                    textcoords="data",
                    xycoords="data",
                    fontsize=pcaMC.fig_Text_Size,
                    horizontalalignment="center", va="center",
                    
                )


        image_name = " Selection."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figdataset.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()

### Spectra vs GC Variation
        pcaGC = PCA( n_components=11 )  # It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        pcaGC.fit(np.transpose(molar_profile))
        GCsc=pcaGC.transform(np.transpose(molar_profile))
        
        figGCdata, axGCdata = plt.subplots(3, 3, figsize=(pcaMC.fig_Size[0],pcaMC.fig_Size[0]))
        axGCdata[0,0] = plt.subplot2grid((22,22), (0, 0), colspan=6, rowspan=6)
        axGCdata[0,1] = plt.subplot2grid((22,22), (0, 8), colspan=6, rowspan=6)
        axGCdata[0,2] = plt.subplot2grid((22,22), (0, 16), colspan=6, rowspan=6)
        axGCdata[1,0] = plt.subplot2grid((22,22), (8, 0), colspan=6, rowspan=6)
        axGCdata[1,1] = plt.subplot2grid((22,22), (8, 8), colspan=6, rowspan=6)
        axGCdata[1,2] = plt.subplot2grid((22,22), (8, 16), colspan=6, rowspan=6)
        axGCdata[2,0] = plt.subplot2grid((22,22), (16, 0), colspan=6, rowspan=6)
        axGCdata[2,1] = plt.subplot2grid((22,22), (16, 8), colspan=6, rowspan=6)
        axGCdata[2,2] = plt.subplot2grid((22,22), (16, 16), colspan=6, rowspan=6)
        figGCdata.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        grps = [[0,0,2,2,4,4,6,6],[False,True,False,True,False,True,False,True],['db','db','or','or','^c','^c','sm','sm']]
#        ixFA = [0,1,2,3,4,5,10,6,7,16,11,8,9,17,12,18,13,14,15]
        for iS in range(8):
            if grps[1][iS]:
                cGrp = lac_Stage
                fillS = 'full'
            else:
                cGrp = ~lac_Stage
                fillS = 'none'
            ix = np.where(np.logical_and(feed==grps[0][iS] , cGrp))[0]
            axGCdata[0,0].plot(pcaMC.component_weight[ix, 0],pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS)
            axGCdata[1,0].plot(-GCsc[ix, 0],-GCsc[ix, 1],grps[2][iS],fillstyle=fillS)
            axGCdata[2,0].plot(-GCsc[ix, 0],pcaMC.component_weight[ix, 0],grps[2][iS],fillstyle=fillS)
        
        lOff = np.round(-(np.max(pcaMC.spectral_loading[1,:])-np.min(pcaMC.spectral_loading[0,:]))*12.5)/10 #offset by a simple number
        axGCdata[0,1].plot(wavelength_axis,pcaMC.spectral_loading[0,:],'k')
        axGCdata[0,1].plot(wavelength_axis,pcaMC.spectral_loading[1,:] + lOff,color=[0.4,0.4,0.4])
        axGCdata[0,1].plot([wavelength_axis[0],wavelength_axis[-1]],[0,0],'--',color=[0.7,0.7,0.7],lw=0.5)
        axGCdata[0,1].plot([wavelength_axis[0],wavelength_axis[-1]],[lOff,lOff],'--',color=[0.7,0.7,0.7],lw=0.5)
#        ixGFA = np.array([0,0,0,0,0,0,1,0,0,2,1,0,0,2,1,2,1,1,1])
        markGFA = [['.','X','*','<'],[[0.5,0.65,0.85],[0.8,0.5,0.65],[0.65,0.85,0.5]],[1/18,1/3,1]]

        GOff = np.round(-(np.max(pcaGC.components_[1,:])-np.min(pcaGC.components_[0,:]))*12.5)/10 #offset by a simple number
        
        for i in range(19):
            marCol = (np.floor(np.array([1,0.4,0.75])/4*(FAprops_N_Olefin[i][0]+1)*255)/255).reshape(1,-1)    
            axGCdata[1,1].scatter(FAprops_N_Carbon[i][0],-pcaGC.components_[0,i],marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol)#)
            axGCdata[1,1].scatter(FAprops_N_Carbon[i][0],-pcaGC.components_[1,i] + GOff,marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol)#)
        axGCdata[1,1].plot([4,18],[0,0],'--',color=[0.7,0.7,0.7])
        axGCdata[1,1].plot([4,18],[GOff,GOff],'--',color=[0.7,0.7,0.7])
        axGCdata[1,1].plot([4,18],[GOff/2,GOff/2],'.-',color=[0.95,0.95,0.95])
        
        
        # crossover models
        GCSpecLoad = np.inner(data,GCsc.T)
        GCSpecLoad = (GCSpecLoad/np.sum(GCSpecLoad**2,axis=0)**0.5)
        lGOff = np.round(-(np.max(-GCSpecLoad[:,1])-np.min(-GCSpecLoad[:,0]))*12.5)/10 #offset by a simple number
        axGCdata[0,2].plot(wavelength_axis,-GCSpecLoad[:,0],'k')
        axGCdata[0,2].plot(wavelength_axis,-GCSpecLoad[:,1] + lGOff,color=[0.4,0.4,0.4])
        axGCdata[0,2].plot([wavelength_axis[0],wavelength_axis[-1]],[0,0],'--',color=[0.7,0.7,0.7],lw=0.5)
        axGCdata[0,2].plot([wavelength_axis[0],wavelength_axis[-1]],[lGOff,lGOff],'--',color=[0.7,0.7,0.7],lw=0.5)
        
        SpecGCLoad = np.inner(molar_profile,pcaMC.component_weight[:,:11].T)
        SpecGCLoad = (SpecGCLoad/np.sum(SpecGCLoad**2,axis=0)**0.5)
        lSOff = np.round(-(np.max(-SpecGCLoad[:,1])-np.min(-SpecGCLoad[:,0]))*12.5)/10 #offset by a simple number
        for i in range(19):
            marCol = (np.floor(np.array([1,0.4,0.75])/4*(FAprops_N_Olefin[i][0]+1)*255)/255).reshape(1,-1)    
            axGCdata[1,2].scatter(FAprops_N_Carbon[i][0],SpecGCLoad[i,0],marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol)#)
            axGCdata[1,2].scatter(FAprops_N_Carbon[i][0],SpecGCLoad[i,1] + lSOff,marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol)#)
        axGCdata[1,2].plot([4,18],[0,0],'--',color=[0.7,0.7,0.7])
        axGCdata[1,2].plot([4,18],[lSOff,lSOff],'--',color=[0.7,0.7,0.7])
        axGCdata[1,2].plot([4,18],[lSOff/2,lSOff/2],'.-',color=[0.95,0.95,0.95])
        
        axGCdata[2,1].plot(-GCSpecLoad[:,0],pcaMC.spectral_loading[0,:],'.k')
        axGCdata[2,1].plot(-GCSpecLoad[:,1],pcaMC.spectral_loading[1,:],'.',color=[0.4,0.4,0.4])
        
        for i in range(19):
            marCol = (np.floor(np.array([0.5,0.2,0.375])/4*(FAprops_N_Olefin[i][0]+1)*255)/255).reshape(1,-1)    
            axGCdata[2,2].scatter(-pcaGC.components_[0,i],SpecGCLoad[i,0],marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol)#)
            axGCdata[2,2].scatter(-pcaGC.components_[1,i],SpecGCLoad[i,1]+lSOff/3,marker=markGFA[0][FAprops_N_Isomer[i][0]],c=marCol*2)#)
        xran = (axGCdata[2,2].get_xlim()[0]*0.95,axGCdata[2,2].get_xlim()[1]*0.95)
        axGCdata[2,2].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[0,:],SpecGCLoad[:,0],1))(xran),'--k',lw=0.5)
        axGCdata[2,2].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[1,:],SpecGCLoad[:,1],1))(xran)+lSOff/3,'--',color=[0.4,0.4,0.4],lw=0.5)
        
        axGCdata[0,0].set_xlabel('t[1]Spectral',labelpad=-1)
        axGCdata[0,0].set_ylabel('t[2]Spectral',labelpad=-1)
        axGCdata[0,0].legend(['0mg G0','0mg G1','2mg','_','4mg','_','6mg'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCdata[1,0].set_xlabel('t[1]GC',labelpad=-1)
        axGCdata[1,0].set_ylabel('t[2]GC',labelpad=-1)
        axGCdata[2,0].set_xlabel('t[1]GC',labelpad=-1)
        axGCdata[2,0].set_ylabel('t[1]Spectral',labelpad=-1)
        
        axGCdata[0,1].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axGCdata[0,1].set_ylabel('Spectral Coefficient',labelpad=-1)
        axGCdata[1,1].set_xlabel('FA chain length',labelpad=-1)
        axGCdata[1,1].set_ylabel('GC Coefficient',labelpad=-1)
        axGCdata[1,1].legend(['_','_','_','Sat','_','_','_','_','_','_','_','_','_','_',
                              '_','Cis1','_','_','_','_','_','Trans1','_','_','_','_','_',
                              '_','_','_','_','_','_','Trans2','_','_','_','Cis3','_','_','CLA'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2,loc=(0.01,0.2))
        axGCdata[2,1].set_xlabel('Cross Coefficient',labelpad=-1)
        axGCdata[2,1].set_ylabel('Spectral Coefficient',labelpad=-1)
        axGCdata[2,1].legend(['PC1','PC2'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2,loc='best')
        
        axGCdata[0,2].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axGCdata[0,2].set_ylabel('Cross Coefficient',labelpad=-1)
        axGCdata[1,2].set_xlabel('FA chain length',labelpad=-1)
        axGCdata[1,2].set_ylabel('Cross Coefficient',labelpad=-1)
        axGCdata[2,2].set_xlabel('GC Coefficient',labelpad=-1)
        axGCdata[2,2].set_ylabel('Cross Coefficient',labelpad=-1)
        
        subLabels = ["a) Spectral Scores","b) Spectral Loadings", "c) Cross GC loadings",
                     "d) GC Scores","e) GC Loadings", "f) Cross Spectral loadings",
                     "g) Spectral vs GC PC2 Scores","h) Spectral Cross Loadings", "i)  GC Cross Loadings"]
        for ax1 in range(3):
            for ax2 in range(3):
                axGCdata[ax1,ax2].annotate(
                    subLabels[ax1*3+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.1, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
        
        axGCdata[2,2].annotate(
            "PC1\n \n \n PC2",
            xy=(0, 0),
            xytext=(xran[0]*0.8,np.poly1d(np.polyfit(-pcaGC.components_[0,:],SpecGCLoad[:,0],1))(xran[0]*0.8)*2),
            textcoords="data",
            xycoords="data",
            horizontalalignment="center",
        )
        
        image_name = " GC crossover plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figGCdata.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        
        pcaFA = npls.nipals(
            X_data=simplified_fatty_acid_spectra["simFA"][201:,:],
            maximum_number_PCs=10,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pcaFA.calc_PCA()
        figFA, axFA = plt.subplots(1, 2, figsize=pcaMC.fig_Size)
        axFA[0] = plt.subplot2grid((1,20), (0, 14), colspan=6, )
        axFA[1] = plt.subplot2grid((1,20), (0, 0), colspan=12, )
        figFA.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        lOff = np.round(-(np.max(pcaFA.spectral_loading[1,:])-np.min(pcaFA.spectral_loading[0,:]))*12.5)/10 #offset by a simple number
        axFA[0].plot(wavelength_axis,pcaFA.spectral_loading[0,:],'k')
        axFA[0].plot(wavelength_axis,pcaFA.spectral_loading[1,:] + lOff,color=[0.4,0.4,0.4])
        axFA[0].plot([wavelength_axis[0],wavelength_axis[-1]],[0,0],'--',color=[0.7,0.7,0.7],lw=0.5)
        axFA[0].plot([wavelength_axis[0],wavelength_axis[-1]],[lOff,lOff],'--',color=[0.7,0.7,0.7],lw=0.5)
        
        axFA[1].plot(pcaFA.component_weight[:,0],pcaFA.component_weight[:,1],'.')
        FAnames = ['4:0','6:0','8:0','10:0','12:0','14:0','14:1c','15:0','16:0','16:1t',
                   '16:1c','17:0','18:0','18:1t','18:1c','18:2t','18:2c','18:3c','CLA']
        offset = 1
        for iFA in range(pcaFA.N_Obs):
            if FAnames[iFA][-1]=='0':
                col = [1,0,0]
            elif FAnames[iFA][-1]=='t':
                col = [0,1,0]
            elif FAnames[iFA][-1]=='c':
                col = [0,0,1]
            else:
                col = [0,0.5,1]
                offset = 0.75
            p1 = pcaFA.component_weight[iFA,0]
            p2 = pcaFA.component_weight[iFA,1]
            axFA[1].annotate(  FAnames[iFA],
                                xy=(p1, p2),
                                xytext=(p1+offset*np.sign(p1), p2+offset*np.sign(p2)),
                                textcoords="data",xycoords="data",
                                fontsize=pcaMC.fig_Text_Size*0.75,
                                horizontalalignment="center",
                                color=col,
                                )
        
        
        axFA[0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axFA[0].set_ylabel('Spectral Coefficient',labelpad=-1)
        
        axFA[1].set_xlabel('t[1]Spectral',labelpad=-1)
        axFA[1].set_ylabel('t[2]Spectral',labelpad=-1)
        
        subLabels = ["a) PC Loadings","b) PC Scores"]
        for ax1 in range(2):
            axFA[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.1, 1.01),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )
        
        image_name = " reference FA PCA."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figFA.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        
### **** END Interpretation ****

### **** START Perturbed Data ****
        SF = 2 # define scale to manipulate the C=C stretch mode
        
        #offset the C=C stretch band but not its variation,to power of SF
        spectraCCpSF = data.copy()
        spectraCCpSF[400:500,:] = (spectraCCpSF[400:500,:].T + (spectraCCpSF[400:500,:]**SF).mean(axis=1)).T
        
        
        #scale the C=C stretch band across its dynamic range,to power of SF
        spectraCCxSF = data.copy()
        spectraCCxSF[400:500,:] =spectraCCxSF[400:500,:]**SF
        CCoffset = np.min(spectraCCxSF[400:500,:],0)
        spectraCCxSF[400:500,:] =((spectraCCxSF[400:500,:]-CCoffset)*SF)+CCoffset #correct the offset
        
        
        
        #plt.plot(FAsim['FAXcal'][[0,0]][0,],spectraCCxSF)
        #plt.ylabel('Intensity')
        #plt.xlabel('Raman Shift cm$^-1$')
        #plt.show()
        
        # no preprocessing.
        PCACCpSF = npls.nipals(
            X_data=spectraCCpSF,
            maximum_number_PCs=nPC,
            preproc="none",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCACCpSF.calc_PCA()
        
        
        # peak offset mean centred.
        PCACCpSFMC = npls.nipals(
            X_data=spectraCCpSF,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCACCpSFMC.calc_PCA()
        
        
        # peak range.
        PCACCxSF = npls.nipals(
            X_data=spectraCCxSF,
            maximum_number_PCs=nPC,
            preproc="none",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCACCxSF.calc_PCA()
        
        PCACCxSFMC = npls.nipals(
            X_data=spectraCCxSF,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCACCxSFMC.calc_PCA()
        
        #  unit scaled version, 
        pcaMCUV = npls.nipals(
            X_data = data,
            maximum_number_PCs = nPC,
            preproc = "MCUV",
            pixel_axis = wavelength_axis,
            spectral_weights = molar_profile,
            min_spectral_values = min_data,
        )
        pcaMCUV.calc_PCA()
        
        pca_1_noiseUV = npls.nipals(
            X_data=data_1_noise,
            maximum_number_PCs=nPC,
            preproc="MCUV",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseUV.calc_PCA()
        
        pca_1_noiseSqrt = npls.nipals(
            X_data=(data_1_noise+100)**.5,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseSqrt.calc_PCA()
        
        pca_1_noiseLn = npls.nipals(
            X_data=np.log(data_1_noise+100),
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseLn.calc_PCA()
        
        # unit scaled version of the dataset manipulated in one peak region
        PCACCxSFMCUV = npls.nipals(
            X_data = spectraCCxSF,
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MCUV",
            pixel_axis = wavelength_axis,
            spectral_weights = molar_profile,
            min_spectral_values = min_data,
        )
        PCACCxSFMCUV.calc_PCA()
        
        # log transformed version of the manipulated data
        PCACCxSFLogMC = npls.nipals(
            X_data = np.log10(spectraCCxSF),
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MC",
            pixel_axis = wavelength_axis,
            spectral_weights = molar_profile,
            min_spectral_values = min_data,
        )
        PCACCxSFLogMC.calc_PCA()
        
### Centring
        figMCplots, axMCplots = plt.subplots(2, 3, figsize=pcaMC.fig_Size)
        axMCplots[0,0] = plt.subplot2grid((13,20), (0, 0), colspan=6, rowspan=6)
        axMCplots[0,1] = plt.subplot2grid((13,20), (0, 7), colspan=6, rowspan=6)
        axMCplots[0,2] = plt.subplot2grid((13,20), (0, 14), colspan=6, rowspan=6)
        axMCplots[1,0] = plt.subplot2grid((13,20), (7, 0), colspan=6, rowspan=6)
        axMCplots[1,1] = plt.subplot2grid((13,20), (7, 7), colspan=6, rowspan=6)
        axMCplots[1,2] = plt.subplot2grid((13,20), (7, 14), colspan=6, rowspan=6)
        figMCplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        axMCplots[0,0].plot( pcaNonMC.pixel_axis , 
                            np.arange(0,-np.ptp(pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaNonMC.spectral_loading[:3,:])/2)
                            +pcaNonMC.spectral_loading[:3,:].T, )
        axMCplots[0,0].legend(("PC1","PC2","PC3"),bbox_to_anchor=(0.1,1),loc='upper right', fontsize='xx-small')
        axMCplots[0,0].plot( pcaNonMC.pixel_axis , 
                            pcaMC.centring/np.ptp(pcaMC.centring)*np.ptp(pcaNonMC.spectral_loading[0,:]), '--c',lw=1)
        axMCplots[0,1].plot( pcaNonMC.pixel_axis , 
                            np.arange(0,-np.ptp(PCACCpSF.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCACCpSF.spectral_loading[:3,:])/2)
                            +PCACCpSF.spectral_loading[:3,:].T, )
        axMCplots[0,1].plot( pcaNonMC.pixel_axis , 
                            PCACCpSFMC.centring/np.ptp(PCACCpSFMC.centring)*np.ptp(PCACCpSF.spectral_loading[0,:].T), '--c',lw=1)
        axMCplots[0,2].plot( pcaNonMC.pixel_axis , 
                            np.arange(0,-np.ptp(PCACCxSF.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCACCxSF.spectral_loading[:3,:])/2)
                            +PCACCxSF.spectral_loading[:3,:].T, )
        axMCplots[0,2].plot( pcaNonMC.pixel_axis , 
                            PCACCpSFMC.centring/np.ptp(PCACCpSFMC.centring)*np.ptp(PCACCxSF.spectral_loading[0,:].T), '--c',lw=1)
        axMCplots[1,0].plot( pcaMC.pixel_axis , 
                            np.arange(0,-np.ptp(pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMC.spectral_loading[:3,:])/2)
                            +pcaMC.spectral_loading[:3,:].T, )
        axMCplots[1,1].plot( pcaNonMC.pixel_axis , 
                            np.arange(0,-np.ptp(PCACCpSFMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCACCpSFMC.spectral_loading[:3,:])/2)
                            +PCACCpSFMC.spectral_loading[:3,:].T, )
        axMCplots[1,2].plot( pcaNonMC.pixel_axis , 
                            np.arange(0,-np.ptp(PCACCxSFMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCACCxSFMC.spectral_loading[:3,:])/2)
                            +PCACCxSFMC.spectral_loading[:3,:].T, )
        
        subLabels = [r"a) $L^\top_{Raw}$",r"b) $L^\top_{PeakOffset}$", r"c)$L^\top_{PeakScale}$",
                     r"d) $L^\top_{Raw}$MC",r"e) $L^\top_{PeakOffset}$MC", r"f) $L^\top_{PeakScale}$MC"]
        for ax1 in range(2):
            for ax2 in range(3):
                axMCplots[ax1,ax2].annotate(
                    subLabels[ax1*3+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.1, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
                if not pcaMC.fig_Show_Values: 
                    axMCplots[ax1,ax2].axis("off")
        
                if pcaMC.fig_Show_Labels:
                    axMCplots[ax1,ax2].set_ylabel(pcaMC.fig_Y_Label)
                    axMCplots[ax1,ax2].set_xlabel(pcaMC.fig_X_Label)
        MAD_MCvspSFMC = np.absolute(PCACCpSFMC.spectral_loading[:2,:]-pcaMC.spectral_loading[:2,:]).mean() #mean absolute difference
        print("Mean Absolute Difference Mean Centered vs Peak Offset Mean Centered: " + str(MAD_MCvspSFMC))
        image_name = " Mean Centring plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figMCplots.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()
        
### Variable-wise Scaling
        figUVplots, axUVplots = plt.subplots(2, 3, figsize=pcaMC.fig_Size)
        axUVplots[0,0] = plt.subplot2grid((13,20), (0, 0), colspan=6, rowspan=6)
        axUVplots[0,1] = plt.subplot2grid((13,20), (0, 7), colspan=6, rowspan=6)
        axUVplots[0,2] = plt.subplot2grid((13,20), (0, 14), colspan=6, rowspan=6)
        axUVplots[1,0] = plt.subplot2grid((13,20), (7, 0), colspan=6, rowspan=6)
        axUVplots[1,1] = plt.subplot2grid((13,20), (7, 7), colspan=6, rowspan=6)
        axUVplots[1,2] = plt.subplot2grid((13,20), (7, 14), colspan=6, rowspan=6)
        figUVplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        axUVplots[0,0].plot( pcaMCUV.pixel_axis , 
                            np.arange(0,-np.ptp(pcaMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMCUV.spectral_loading[:3,:])/2)
                            +pcaMCUV.spectral_loading[:3,:].T, )
        axUVplots[0,0].legend(("PC1","PC2","PC3"),bbox_to_anchor=(0.1,0.9),loc='upper right', fontsize='xx-small')
        axUVplots[0,1].plot( PCACCxSFMCUV.pixel_axis , 
                            np.arange(0,-np.ptp(PCACCxSFMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCACCxSFMCUV.spectral_loading[:3,:])/2)
                            +PCACCxSFMCUV.spectral_loading[:3,:].T, )
        MAD_MCUVvsxSFMCUV = np.absolute(PCACCxSFMCUV.spectral_loading[:2,:]-pcaMCUV.spectral_loading[:2,:]).mean() #mean absolute difference
        print("Mean Absolute Deviation Mean Centered Unit Variance vs Multiplicative Scaled Mean Centered Unit Scaled: " + str(MAD_MCUVvsxSFMCUV))
        print("orders of magnitude difference in MAD for correction of multiplicative vs additive effects: " + str(np.floor(np.log10(MAD_MCUVvsxSFMCUV/MAD_MCvspSFMC))))
        axUVplots[0,2].plot( pca_1_noiseSqrt.pixel_axis , 
                            np.arange(0,-np.ptp(pca_1_noiseSqrt.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noiseSqrt.spectral_loading[:3,:])/2)
                            +pca_1_noiseSqrt.spectral_loading[:3,:].T, )
        axUVplots[1,0].plot( pca_1_noise.pixel_axis , 
                            np.arange(0,-np.ptp(pca_1_noise.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noise.spectral_loading[:3,:])/2)
                            +pca_1_noise.spectral_loading[:3,:].T, )
        axUVplots[1,1].plot( pca_1_noiseUV.pixel_axis , 
                            np.arange(0,-np.ptp(pca_1_noiseUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noiseUV.spectral_loading[:3,:])/2)
                            +pca_1_noiseUV.spectral_loading[:3,:].T, )
        axUVplots[1,2].plot( pca_1_noiseLn.pixel_axis , 
                            np.arange(0,-np.ptp(pca_1_noiseLn.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noiseLn.spectral_loading[:3,:])/2)
                            +pca_1_noiseLn.spectral_loading[:3,:].T, )
        
        subLabels = [r"a) $L^\top_{Raw}$ MCUV",r"b) $L^\top_{PeakScale}$ MCUV", r"c)$L^\top_{\sqrt{Noisy}}$ MC",
                     r"d) $L^\top_{Noisy}$ MC",r"e) $L^\top_{Noisy}$ MCUV", r"f) $L^\top_{Ln(Noisy)}$ MC"]
        #find out how to display a square root rather than calculate one
        for ax1 in range(2):
            for ax2 in range(3):
                axUVplots[ax1,ax2].annotate(
                    subLabels[ax1*3+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.1, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
                if not pcaMC.fig_Show_Values: 
                    axUVplots[ax1,ax2].axis("off")
        
                if pcaMC.fig_Show_Labels:
                    axUVplots[ax1,ax2].set_ylabel(pcaMC.fig_Y_Label)
                    axUVplots[ax1,ax2].set_xlabel(pcaMC.fig_X_Label)
        
        image_name = " Scaling plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figUVplots.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        plt.close()

### Observation-wise Scaling
        
        # create variation in intensity, then normalise to carbonyl, CHx and vector norm
        ints = stat.skewnorm.rvs(5,loc=1,scale=0.2,size=data.shape[1],random_state=6734325)
        data_Int = data*ints
        CO_Int = np.sum(data_Int[535:553,:],axis=0)
        CHx_Int = np.sum(data_Int[227:265,:],axis=0)
        Vec_Int = np.sum(data_Int**2,axis=0)**0.5
        PCA_Int = npls.nipals(
            X_data=data_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCA_Int.calc_PCA()
        PCA_IntCO = npls.nipals(
            X_data=data_Int/CO_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCA_IntCO.calc_PCA()
        PCA_IntCHx = npls.nipals(
            X_data=data_Int/CHx_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCA_IntCHx.calc_PCA()
        PCA_IntVec = npls.nipals(
            X_data=data_Int/Vec_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        PCA_IntVec.calc_PCA()
        
        figIntplots, axIntplots = plt.subplots(1, 4, figsize=pcaMC.fig_Size)
        axIntplots[0] = plt.subplot2grid((1,27), (0, 0), colspan=6)
        axIntplots[1] = plt.subplot2grid((1,27), (0, 7), colspan=6)
        axIntplots[2] = plt.subplot2grid((1,27), (0, 14), colspan=6)
        axIntplots[3] = plt.subplot2grid((1,27), (0, 21), colspan=6)
        figIntplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        axIntplots[0].plot( PCA_Int.pixel_axis , 
                            np.arange(0,-np.ptp(PCA_Int.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_Int.spectral_loading[:3,:])/2)
                            +PCA_Int.spectral_loading[:3,:].T, )
        axIntplots[0].add_artist(patches.Ellipse((PCA_Int.pixel_axis[244],
                                          PCA_Int.spectral_loading[0,244]/2),
                                         width=100,
                                         height=np.abs(PCA_Int.spectral_loading[0,244])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[1].plot( PCA_IntCO.pixel_axis , 
                            np.arange(0,-np.ptp(PCA_IntCO.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntCO.spectral_loading[:3,:])/2)
                            +PCA_IntCO.spectral_loading[:3,:].T, )
        axIntplots[1].add_artist(patches.Ellipse((PCA_IntCO.pixel_axis[244],
                                          PCA_IntCO.spectral_loading[0,244]/2),
                                         width=100,
                                         height=np.abs(PCA_IntCO.spectral_loading[0,244])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[1].add_artist(patches.Rectangle((PCA_IntCO.pixel_axis[535],np.min(PCA_IntCO.spectral_loading[0,535:553])*1.25),
                                         width=18,
                                         height=np.ptp(PCA_IntCO.spectral_loading[0,535:553])*1.5,
                                         facecolor='none', edgecolor='g',linestyle='--'))
        axIntplots[2].plot( PCA_IntCHx.pixel_axis , 
                            np.arange(0,-np.ptp(PCA_IntCHx.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntCHx.spectral_loading[:3,:])/2)
                            +PCA_IntCHx.spectral_loading[:3,:].T, )
        axIntplots[2].add_artist(patches.Ellipse((PCA_IntCHx.pixel_axis[244],
                                          PCA_IntCHx.spectral_loading[0,244]/2),
                                         width=100,
                                         height=np.abs(PCA_IntCHx.spectral_loading[0,244])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[2].add_artist(patches.Rectangle((PCA_IntCHx.pixel_axis[227],np.min(PCA_IntCHx.spectral_loading[0,227:265])*1.1),
                                         width=38,
                                         height=np.ptp(PCA_IntCHx.spectral_loading[0,227:265])*1.2,
                                         facecolor='none', edgecolor='g',linestyle='--'))
        axIntplots[3].plot( PCA_IntVec.pixel_axis , 
                            np.arange(0,-np.ptp(PCA_IntVec.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntVec.spectral_loading[:3,:])/2)
                            +PCA_IntVec.spectral_loading[:3,:].T, )
        axIntplots[3].add_artist(patches.Ellipse((PCA_IntVec.pixel_axis[244],
                                          PCA_IntVec.spectral_loading[0,244]/2),
                                         width=100,
                                         height=np.abs(PCA_IntVec.spectral_loading[0,244])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[3].add_artist(patches.Rectangle((PCA_IntVec.pixel_axis[0],np.min(PCA_IntVec.spectral_loading[0,:])*1.05),
                                         width=PCA_IntVec.N_Vars,
                                         height=np.ptp(PCA_IntCHx.spectral_loading[0,:])*1.1,
                                         facecolor='none', edgecolor='g',linestyle='--'))
        
        subLabels = [r"a) $L^\top_{Int}$ MC Unnorm", r"b) $L^\top_{Int}$ MC/C=O",
                     r"c) $L^\top_{Int}$ MC/CH$_x$", r"d)$L^\top_{Int}$ MC/Norm"]

        for ax1 in range(axIntplots.shape[0]):
            axIntplots[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.1, 0.98),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )
            if not pcaMC.fig_Show_Values: 
                axIntplots[ax1].axis("off")
        
            if pcaMC.fig_Show_Labels:
                axIntplots[ax1].set_ylabel(pcaMC.fig_Y_Label)
                axIntplots[ax1].set_xlabel(pcaMC.fig_X_Label)
        
        image_name = " Normalisation plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figIntplots.savefig(full_path, 
                         dpi=pcaMC.fig_Resolution)
        
        plt.close()
### **** END Perturbed Data ****

### ******      END CLASS      ******
        return
