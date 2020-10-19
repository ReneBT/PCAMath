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
            os.path.join(data_folder / "FA profile.mat"), struct_as_record=False
        )
# Gas Chromatograph (GC) data is modelled based on Beattie et al. Lipids 2004 Vol 39 (9):897-906
# it is reconstructed with 4 underlying factors
        simplified_fatty_acid_spectra = sio.loadmat(data_folder / "FA spectra.mat", struct_as_record=False)
# simplified_fatty_acid_spectra are simplified spectra of fatty acid methyl esters built from the properties described in
# Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        wavelength_axis = simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][201:,]
        min_spectral_values = np.tile(
            np.min(simplified_fatty_acid_spectra["simFA"][201:,:], axis=1), (np.shape(simplified_fatty_acid_spectra["simFA"][201:,:])[1], 1)
        )
# convert the mass base profile provided into a molar profile
        molar_profile = GC_data["GCprofile"] / simplified_fatty_acid_spectra["FAproperties"][0, 0].MolarMass
        molar_profile = 100.0 * molar_profile / sum(molar_profile)
        sam_codes = GC_data["sample_ID"] 
        lac_Stage = np.empty(np.size(sam_codes),dtype=bool)
        feed = np.empty(np.size(sam_codes))
        week = np.empty(np.size(sam_codes))
        for iSam in range(sam_codes.shape[1]):
            lac_Stage[iSam] = sam_codes[0,iSam][0][0]=='B'
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
            maximum_number_PCs=6,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=molar_profile,
            min_spectral_values=min_data,
        )
        pcaMC.calc_PCA()

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
        plt.close()


### ***   Start  Interpretation   ***

### Biplot

        figbiplots, axbiplots = plt.subplots(2, 3, figsize=[9,5.3])
        axbiplots[0,0] = plt.subplot2grid((14,22), (0, 0), colspan=6, rowspan=6)
        axbiplots[0,1] = plt.subplot2grid((14,22), (0, 8), colspan=6, rowspan=6)
        axbiplots[0,2] = plt.subplot2grid((14,22), (0, 16), colspan=6, rowspan=6)
        axbiplots[1,0] = plt.subplot2grid((14,22), (8, 0), colspan=6, rowspan=6)
        axbiplots[1,1] = plt.subplot2grid((14,22), (8, 8), colspan=6, rowspan=6)
        axbiplots[1,2] = plt.subplot2grid((14,22), (8, 16), colspan=6, rowspan=6)
        figbiplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
        
        grps = [[0,0,1,1,2,2,3,3],[False,True,False,True,False,True,False,True],['db','db','or','or','^c','^c','sm','sm']]
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
        plt.close()

### **** END Interpretation ****

### ******      END CLASS      ******
        return
