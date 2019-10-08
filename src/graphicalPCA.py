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
        # Read in simulated fatty acid spectra and associated experimental concentration data
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
        )  # /np.mean(simplified_fatty_acid_spectra['simFA'],axis=1)

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

        # Now we calculate PCA, first with the full FA covariance, comparing NIPALS and built in function.

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


        ###################            START DSLTmainEqn              #######################
        # FIGURE for the main PCA equation
        pca_full_covariance.figure_DSLT("DSLT")
        ###################             END  DSLTmainEqn              #######################

        ###################            START SLTDscoreEqn             #######################
        pca_full_covariance.figure_DSLT("SLTD")
        ###################             END  SLTDscoreEqn             #######################

        ###################            START LTSDscoreEqn             #######################
        pca_full_covariance.figure_DSLT("LTSD")
        ###################             END  LTSDscoreEqn             #######################

### NOTE THIS CODE IS CURRENTLY RETAINED TO PREVENT BREAKING DOWNSTREAM FIGURE PLOTS UNTIL ALL ARE CONVERTED TO CLASS FUNCTIONS ####
        data4plot = np.empty([spectra_full_covariance .shape[0],10])
        dataSq4plot = np.empty([spectra_full_covariance .shape[0],10])
        REigenvectors4plot = np.empty([spectra_full_covariance .shape[0],5])
        LEigenvectors4plot = np.empty([10,5])
        SSQ = np.sum(pca_full_covariance .X**2,1)
        for iDat in range(10):
            data4plot[:,iDat] = pca_full_covariance .X[:,iDat]+iDat*16
            dataSq4plot[:,iDat] = pca_full_covariance .X[:,iDat]**2+iDat*1000
            LEigenvectors4plot[iDat,:] = pca_full_covariance .LEigenvector[iDat,0:5]+iDat*40
        for iDat in range(5):
            REigenvectors4plot[:,iDat] = pca_full_covariance .REigenvector[iDat,:]+1-iDat/5


        ###################         START sldiLEigenvectorEqn         #######################
        # FIGURE for the ith LEigenvector equation si = lixD
        figsldi, axsldi = plt.subplots(1, 5, figsize=(8, 8))
        axsldi[0] = plt.subplot2grid((1, 20), (0, 0), colspan=8)
        axsldi[1] = plt.subplot2grid((1, 20), (0, 8), colspan=1)
        axsldi[2] = plt.subplot2grid((1, 20), (0, 9), colspan=8)
        axsldi[3] = plt.subplot2grid((1, 20), (0, 17), colspan=1)
        axsldi[4] = plt.subplot2grid((1, 20), (0, 18), colspan=2)

        iPC = 1  # the ith PC to plot
        iSamMin = np.argmin(pca_full_covariance.LEigenvector[:, iPC - 1])
        iSamMax = np.argmax(pca_full_covariance.LEigenvector[:, iPC - 1])
        iSamZer = np.argmin(
            np.abs(pca_full_covariance.LEigenvector[:, iPC - 1])
        )  # Sam = 43 #the ith sample to plot
        sf_iSam = np.mean(
            [
                sum(pca_full_covariance.X[:, iSamMin] ** 2) ** 0.5,
                sum(pca_full_covariance.X[:, iSamMax] ** 2) ** 0.5,
                sum(pca_full_covariance.X[:, iSamZer] ** 2) ** 0.5,
            ]
        )  # use samescaling factor to preserve relative intensity
        offset = np.max(pca_full_covariance.REigenvector[iPC - 1, :]) - np.min(
            pca_full_covariance.REigenvector[iPC - 1, :]
        )  # offset for clarity
        axsldi[0].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.REigenvector[iPC - 1, :] + offset * 1.25,
            "k",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamMax] / sf_iSam + offset / 4,
            "r",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamZer] / sf_iSam,
            "b",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamMin] / sf_iSam - offset / 4,
            "g",
        )
        axsldi[0].legend(("$pc_i$", "$d_{max}$", "$d_0$", "$d_{min}$"))
        temp = REigenvectors4plot[:, iPC - 1] * pca_full_covariance.X[:, iSamZer]
        offsetProd = np.max(temp) - np.min(temp)
        axsldi[2].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            REigenvectors4plot[:, iPC - 1] * pca_full_covariance.X[:, iSamMax] + offsetProd,
            "r",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            REigenvectors4plot[:, iPC - 1] * pca_full_covariance.X[:, iSamZer],
            "b",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            REigenvectors4plot[:, iPC - 1] * pca_full_covariance.X[:, iSamMin] - offsetProd,
            "g",
        )

        PCilims = np.tile(
            np.array(
                [
                    np.average(pca_full_covariance.LEigenvector[:, iPC - 1])
                    - 1.96 * np.std(pca_full_covariance.LEigenvector[:, iPC - 1]),
                    np.average(pca_full_covariance.LEigenvector[:, iPC - 1]),
                    np.average(pca_full_covariance.LEigenvector[:, iPC - 1])
                    + 1.96 * np.std(pca_full_covariance.LEigenvector[:, iPC - 1]),
                ]
            ),
            (2, 1),
        )
        axsldi[4].plot(
            [0, 10],
            PCilims,
            "k--",
            5,
            pca_full_covariance.LEigenvector[iSamMax, iPC - 1],
            "r.",
            5,
            pca_full_covariance.LEigenvector[iSamZer, iPC - 1],
            "b.",
            5,
            pca_full_covariance.LEigenvector[iSamMin, iPC - 1],
            "g.",
            markersize=10,
        )
        ylimLEV = (
            np.abs(
                [
                    pca_full_covariance.LEigenvector[:, iPC - 1].min(),
                    pca_full_covariance.LEigenvector[:, iPC - 1].max(),
                ]
            ).max()
            * 1.05
        )
        axsldi[4].set_ylim([-ylimLEV, ylimLEV])
        axsldi[1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldi[1].annotate(
            r"$pc_i \times d_i$",
            xy=(0.5, 0.5),
            xytext=(0.5, 0.52),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axsldi[3].annotate(
            "$\Sigma _{v=1}^{v=p}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.52),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axsldi[3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=18,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldi[4].annotate(
            "$U95CI$",
            xy=(5, PCilims[0, 2]),
            xytext=(10, PCilims[0, 2]),
            xycoords="data",
            textcoords="data",
            fontsize=12,
            horizontalalignment="left",
        )
        axsldi[4].annotate(
            "$\overline{S_{i}}$",
            xy=(0, 0.9),
            xytext=(1, 0.49),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="left",
        )
        axsldi[4].annotate(
            "$L95CI$",
            xy=(5, PCilims[0, 0]),
            xytext=(10, PCilims[0, 0]),
            xycoords="data",
            textcoords="data",
            fontsize=12,
            horizontalalignment="left",
        )
        for iax in range(5):
            axsldi[iax].axis("off")

        figsldi.savefig(images_folder / "sldiLEigenvectorEqn.png", dpi=300)
        plt.close()
        ###################          END  sldiLEigenvectorEqn         #######################

        ###################          Start  sldiResidual         #######################

        figsldiRes, axsldiRes = plt.subplots(1, 3, figsize=(8, 8))
        axsldiRes[0] = plt.subplot2grid((1, 17), (0, 0), colspan=8)
        axsldiRes[1] = plt.subplot2grid((1, 17), (0, 8), colspan=1)
        axsldiRes[2] = plt.subplot2grid((1, 17), (0, 9), colspan=8)

        iSamResMax = pca_full_covariance.X[:, iSamMax] - np.inner(
            pca_full_covariance.LEigenvector[iSamMax, iPC - 1],
            pca_full_covariance.REigenvector[iPC - 1, :],
        )
        iSamResZer = pca_full_covariance.X[:, iSamZer] - np.inner(
            pca_full_covariance.LEigenvector[iSamZer, iPC - 1],
            pca_full_covariance.REigenvector[iPC - 1, :],
        )
        iSamResMin = pca_full_covariance.X[:, iSamMin] - np.inner(
            pca_full_covariance.LEigenvector[iSamMin, iPC - 1],
            pca_full_covariance.REigenvector[iPC - 1, :],
        )
        offsetRes = np.max(iSamResZer) - np.min(iSamResZer)

        axsldiRes[0].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamMax] / sf_iSam + offset / 4,
            "r",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamZer] / sf_iSam,
            "b",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            pca_full_covariance.X[:, iSamMin] / sf_iSam - offset / 4,
            "g",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.inner(
                pca_full_covariance.LEigenvector[iSamMax, iPC - 1],
                pca_full_covariance.REigenvector[iPC - 1, :],
            )
            / sf_iSam
            + offset / 4,
            "k--",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.inner(
                pca_full_covariance.LEigenvector[iSamZer, iPC - 1],
                pca_full_covariance.REigenvector[iPC - 1, :],
            )
            / sf_iSam,
            "k--",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.inner(
                pca_full_covariance.LEigenvector[iSamMin, iPC - 1],
                pca_full_covariance.REigenvector[iPC - 1, :],
            )
            / sf_iSam
            - offset / 4,
            "k--",
        )
        axsldiRes[0].legend(("$d_{max}$", "$d_0$", "$d_{min}$", "$pc_i*d_{j}$"))

        axsldiRes[2].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            iSamResMax + offsetRes / 2,
            "r",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            iSamResZer,
            "b",
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            iSamResMin - offsetRes / 2,
            "g",
        )

        axsldiRes[1].annotate(
            "",
            xy=(1, 0.6),
            xytext=(0, 0.6),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRes[1].annotate(
            r"$d_i-(pc_i \times d_i)$",
            xy=(0.5, 0.5),
            xytext=(0.5, 0.62),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        for iax in range(3):
            axsldiRes[iax].axis("off")

        figsldiRes.savefig(images_folder / "sldiResLEigenvectorEqn.png", dpi=300)
        plt.close()

        iSamResCorr = np.corrcoef(iSamResMin, iSamResZer)[0, 1] ** 2
        ###################              END sldiResidual            #######################

 
        ###################        START lsdiLEigenvectorEqn          #######################
        # FIGURE for the ith REigenvector equation Li = 1/Si*D
        figlsdi, axlsdi = plt.subplots(1, 5, figsize=(8, 8))
        axlsdi[0] = plt.subplot2grid((1, 20), (0, 0), colspan=6)
        axlsdi[1] = plt.subplot2grid((1, 20), (0, 6), colspan=1)
        axlsdi[2] = plt.subplot2grid((1, 20), (0, 7), colspan=6)
        axlsdi[3] = plt.subplot2grid((1, 20), (0, 13), colspan=1)
        axlsdi[4] = plt.subplot2grid((1, 20), (0, 14), colspan=6)

        scsf = (
            PCilims[0, 2] - PCilims[0, 0]
        ) / 100  # scale range of LEigenvectors to display beside data
        iPC = 1
        axlsdi[0].plot(
            np.tile(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0] - 100, (9, 1)),
            pca_full_covariance.LEigenvector[0:9, iPC - 1],
            ".",
        )
        axlsdi[0].plot(
            [simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0] - 110, simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0] - 90],
            np.tile(PCilims[0, 1], (2, 1)),
            "k",
        )  # mean score for ith PC
        # TODO need different colour per sample
        axlsdi[0].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], 1.8 * data4plot - 250)
        axlsdi[0].annotate(
            "$s_i$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0] - 110, 0.9),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0] - 110, 0.9),
            textcoords="data",
            fontsize=12,
            horizontalalignment="left",
        )
        axlsdi[2].plot(pca_full_covariance.LEigenvector[0:9, iPC - 1] * pca_full_covariance.X[:, 0:9])
        axlsdi[2].annotate(
            r"$s_i \times d_i$",
            xy=(0.1, 0.9),
            xytext=(0.6, 0.9),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axlsdi[4].plot(pca_full_covariance.REigenvector[iPC - 1, :])
        axlsdi[4].annotate(
            "$L$",
            xy=(0.1, 0.9),
            xytext=(0.5, 0.9),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )

        axlsdi[1].annotate(
            r"$\cdot$",
            xy=(0, 0.5),
            xytext=(0.5, 0.52),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axlsdi[1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axlsdi[3].annotate(
            "$\Sigma _{o=1}^{o=n}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.52),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axlsdi[3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        for iax in range(5):
            axlsdi[iax].axis("off")

        figlsdi.savefig(images_folder / "lsdiLEigenvectorEqn.png", dpi=300)
        plt.close()
        ###################              END lsdiLEigenvectorEqn             #######################

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
        xview = range(625, 685)  # zoom in on region to check in detail for changes
        pca_full_covarianceNMC.figure_lpniCommonSignalScalingFactors(nPC, xview)
        ###################         END lpniCommonSignalScalingFactors           #######################

        ###################                  START DTDscoreEqn                  #######################
        # FIGURE showing how the inner product of the data forms the sum of squares
        figDTD, axDTD = plt.subplots(1, 5, figsize=(8, 8))
        axDTD[0] = plt.subplot2grid((1, 20), (0, 0), colspan=6)
        axDTD[1] = plt.subplot2grid((1, 20), (0, 6), colspan=1)
        axDTD[2] = plt.subplot2grid((1, 20), (0, 7), colspan=6)
        axDTD[3] = plt.subplot2grid((1, 20), (0, 13), colspan=1)
        axDTD[4] = plt.subplot2grid((1, 20), (0, 14), colspan=6)

        axDTD[0].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], data4plot)
        axDTD[2].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], dataSq4plot)
        axDTD[2].annotate(
            "$d_i*d_i$",
            xy=(0.1, 0.9),
            xytext=(0.6, 0.9),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )

        axDTD[4].plot(SSQ)
        axDTD[4].annotate(
            "Sum of Squares",
            xy=(0.1, 0.9),
            xytext=(0.5, 0.9),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )

        axDTD[1].annotate(
            "$d_i^2$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axDTD[1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axDTD[3].annotate(
            "$\Sigma _{o=1}^{o=O}(d_i^2)$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
        )
        axDTD[3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        for iax in range(5):
            axDTD[iax].axis("off")
        figDTD.savefig(images_folder / "DTDscoreEqn.png", dpi=300)
        plt.close()
        ###################                  END DTDscoreEqn                  #######################

        ###################                  START D2DwscoreEqn               #######################
        # FIGURE for the illustration of the NIPALs algorithm, with the aim of iteratively calculating
        # each PCA to minimise the explantion of the sum of squares
        figD2Dw, axD2Dw = plt.subplots(3, 5, figsize=(8, 8))
        axD2Dw[0, 0] = plt.subplot2grid((13, 20), (0, 0), colspan=6, rowspan=6)
        axD2Dw[0, 1] = plt.subplot2grid((13, 20), (0, 6), colspan=1, rowspan=6)
        axD2Dw[0, 2] = plt.subplot2grid((13, 20), (0, 7), colspan=6, rowspan=6)
        axD2Dw[0, 3] = plt.subplot2grid((13, 20), (0, 13), colspan=1, rowspan=6)
        axD2Dw[0, 4] = plt.subplot2grid((13, 20), (0, 14), colspan=6, rowspan=6)
        axD2Dw[1, 0] = plt.subplot2grid((13, 20), (7, 0), colspan=7, rowspan=1)
        axD2Dw[1, 1] = plt.subplot2grid((13, 20), (7, 7), colspan=7, rowspan=1)
        axD2Dw[1, 2] = plt.subplot2grid((13, 20), (7, 14), colspan=6, rowspan=1)
        axD2Dw[2, 0] = plt.subplot2grid((13, 20), (8, 0), colspan=6, rowspan=6)
        axD2Dw[2, 1] = plt.subplot2grid((13, 20), (8, 6), colspan=1, rowspan=6)
        axD2Dw[2, 2] = plt.subplot2grid((13, 20), (8, 7), colspan=6, rowspan=6)
        axD2Dw[2, 3] = plt.subplot2grid((13, 20), (8, 13), colspan=1, rowspan=6)
        axD2Dw[2, 4] = plt.subplot2grid((13, 20), (8, 14), colspan=6, rowspan=6)

        axD2Dw[0, 0].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.r[0])
        ylims0_0 = np.max(np.abs(axD2Dw[0, 0].get_ylim()))
        axD2Dw[0, 0].set_ylim(
            -ylims0_0, ylims0_0
        )  # tie the y limits so scales directly comparable
        axD2Dw[0, 0].annotate(
            "a) $D_{-\mu}=R_0$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 0].get_ylim()[1]),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 0].get_ylim()[1]),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        axD2Dw[0, 1].annotate(
            "$\widehat{\Sigma(R_0^2)}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[0, 1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        axD2Dw[0, 1].annotate(
            "$j=0$",
            xy=(0, 0.5),
            xytext=(0.5, 0.45),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[0, 2].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.pc[1][:, 0], "m")
        axD2Dw[0, 2].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.pc[0][:, 0])
        axD2Dw[0, 2].annotate(
            "b) $\widehat{SS_R}$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 2].get_ylim()[1]),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 2].get_ylim()[1]),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        axD2Dw[0, 3].annotate(
            "$R_{i-1}/\widehat{SS}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[0, 3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[0, 3].annotate(
            "$i=i+1$",
            xy=(0, 0.5),
            xytext=(0.5, 0.45),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[0, 3].annotate(
            "$j=j+1$",
            xy=(0, 0.5),
            xytext=(0.5, 0.40),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[0, 4].plot(pca_full_covariance.w[1][0, :], ".m")
        axD2Dw[0, 4].plot(pca_full_covariance.w[0][0, :] / 10, ".")
        axD2Dw[0, 4].plot(pca_full_covariance.w[0][1, :], ".c")
        axD2Dw[0, 4].annotate(
            "c) $S_i^j$",
            xy=(8, axD2Dw[0, 4].get_ylim()[1]),
            xytext=(8, axD2Dw[0, 4].get_ylim()[1]),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        axD2Dw[1, 2].annotate(
            "",
            xy=(0.5, 0),
            xytext=(0.5, 1),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[1, 2].annotate(
            "$D_{-\mu}/S_i^j$",
            xy=(0.55, 0.5),
            xytext=(0.55, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            rotation=90,
        )
        axD2Dw[2, 4].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.pc[1][:, 1], "m")
        axD2Dw[2, 4].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.pc[0][:, 1])
        axD2Dw[2, 4].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.pc[0][:, 2], "c")
        ylims2_4 = np.max(np.abs(axD2Dw[2, 4].get_ylim()))
        axD2Dw[2, 4].set_ylim(
            -ylims2_4, ylims2_4
        )  # tie the y limits so scales directly comparable
        axD2Dw[2, 4].annotate(
            "d) $L_i^{Tj}$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[2, 4].get_ylim()[0]),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[2, 4].get_ylim()[0]),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        axD2Dw[2, 3].annotate(
            "",
            xy=(0, 0.5),
            xytext=(1, 0.5),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[2, 3].annotate(
            "$|L_i^{jT}-L_i^{(j-1)T}|$",
            xy=(0.5, 0.55),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[2, 2].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.abs(pca_full_covariance.pc[1][:, 1] - pca_full_covariance.pc[1][:, 0]),
            "m",
        )
        axD2Dw[2, 2].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.abs(pca_full_covariance.pc[0][:, 1] - pca_full_covariance.pc[0][:, 0]),
        )
        ylims2_2 = np.max(np.abs(axD2Dw[2, 2].get_ylim())) * 1.1
        axD2Dw[2, 2].plot(
            simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,],
            np.abs(pca_full_covariance.pc[0][:, 2] - pca_full_covariance.pc[0][:, 1]),
            "c",
        )
        axD2Dw[2, 2].set_ylim([0 - ylims2_2 * 0.1, ylims2_2])
        axD2Dw[2, 2].annotate(
            "e) Iteration Change in $L^T$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], 0 - ylims2_2 * 0.1),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], 0 - ylims2_2 * 0.1),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        axD2Dw[2, 2].annotate(
            "$\Sigma|L_i^{jT}-L_i^{(j-1)T}|<Tol$ OR $j=max\_j$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][200], ylims2_2 * 0.9),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], ylims2_2 * 0.87),
            textcoords="data",
            fontsize=8,
            horizontalalignment="center",
        )
        axD2Dw[2, 2].annotate(
            "$True$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0], ylims2_2 * 0.85),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][50], ylims2_2 * 0.79),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
            color="g",
        )
        axD2Dw[2, 2].annotate(
            "$False$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][400], ylims2_2 * 0.99),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][450], ylims2_2 * 0.95),
            textcoords="data",
            fontsize=8,
            horizontalalignment="right",
            color="r",
        )
        con = ConnectionPatch(
            xyA=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][100], ylims2_2 * 0.79),
            xyB=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][-1], ylims0_0 * 0.1),
            coordsA="data",
            coordsB="data",
            axesA=axD2Dw[2, 2],
            axesB=axD2Dw[2, 0],
            arrowstyle="->",
        )
        axD2Dw[2, 2].add_artist(con)
        con2 = ConnectionPatch(
            xyA=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][450], ylims2_2),
            xyB=(axD2Dw[0, 4].get_xlim()[0], axD2Dw[0, 4].get_ylim()[0]),
            coordsA="data",
            coordsB="data",
            axesA=axD2Dw[2, 2],
            axesB=axD2Dw[0, 4],
            arrowstyle="->",
        )
        axD2Dw[2, 2].add_artist(con2)
        axD2Dw[1, 1].annotate(
            "$R_{i-1}^T*L_i^j}$",
            xy=(0, 0.5),
            xytext=(0.7, 1),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="right",
            rotation=42,
        )
        axD2Dw[1, 1].annotate(
            "$j=j+1$",
            xy=(0, 0.5),
            xytext=(0.75, 0.95),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            rotation=42,
        )
        axD2Dw[2, 1].annotate(
            "$R_{i-1}-S_{i}*L_{i}^T$",
            xy=(0.1, 0.9),
            xytext=(0.75, 0.82),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            rotation=40,
        )
        axD2Dw[2, 0].plot(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,], pca_full_covariance.r[1])
        axD2Dw[2, 0].set_ylim(
            -ylims0_0, ylims0_0
        )  # tie the y limits so scales directly comparable
        axD2Dw[2, 0].annotate(
            "f) $R_i$",
            xy=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 0].get_ylim()[0]),
            xytext=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][10], axD2Dw[0, 0].get_ylim()[0]),
            textcoords="data",
            fontsize=8,
            horizontalalignment="left",
        )
        con3 = ConnectionPatch(
            xyA=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][450], np.max(pca_full_covariance.r[1] * 2)),
            xyB=(simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,][0], 0),
            coordsA="data",
            coordsB="data",
            axesA=axD2Dw[2, 0],
            axesB=axD2Dw[0, 2],
            arrowstyle="->",
        )
        axD2Dw[2, 0].add_artist(con3)
        axD2Dw[1, 0].annotate(
            "$\widehat{\Sigma(R_i^2)}$",
            xy=(0, 0.5),
            xytext=(0.75, 0.9),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            rotation=55,
        )
        axD2Dw[1, 0].annotate(
            "$j=0$",
            xy=(0, 0.5),
            xytext=(0.85, 0.5),
            textcoords="axes fraction",
            fontsize=8,
            horizontalalignment="center",
            rotation=55,
        )
        for iax in range(5):
            axD2Dw[0, iax].axis("off")
            axD2Dw[2, iax].axis("off")
            if iax < 3:
                axD2Dw[1, iax].axis("off")
        figD2Dw.subplots_adjust(wspace=0, hspace=0)
        figD2Dw.savefig(images_folder / "D2DwscoreEqn.png", dpi=300)
        plt.close()
        ###################                  END D2DwscoreEqn                  #######################
        return
