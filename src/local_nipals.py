import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import transforms
from pathlib import Path

# This expects to be called inside the jupyter project folder structure.
from src.file_locations import data_folder,images_folder


class nipals:
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

    def __init__(
        self,
        X_data,
        maximum_number_PCs,
        maximum_iterations_PCs=20,
        iteration_tolerance=0.00001,
        preproc="None",
        pixel_axis="None",
        spectral_weights=None,
        min_spectral_values=None,
    ):
        # requires a data matrX_data and optional additional settings
        # X_data      X, main data matrX_data as specified in pseudocode
        # maximum_number_PCs    is max_i, desired maximum number of PCs
        # maximum_iterations_PCs   is max_j, maximum iterations on each PC
        # iteration_tolerance    is tol, acceptable tolerance for difference between j and j-1 estimates of the PCs
        # preproc is defining preprocessing steps desired prior to PCA. Current options are
        #                     mean centering ('MC')
        #                     median centering ('MdnC')
        #                     scaling to unit variance ('UV')
        #                     scaling to range ('MinMax')
        #                     scaling to largest absolute value ('MaxAbs')
        #                     combining these (e.g.'MCUV').
        #            if other types are desired please handle these prior to calling this class
        #            Note that only one centering method and one scaling method will be implmented. If more are present then the
        #            implmented one will be the first in the above list.
        # pixel_axis    is a vector of values for each pixel in the x-axis or a tuple specifying a fX_dataed interval spacing
        #         (2 values for unit spacing, 3 values for non-unit spacing)
        # spectral_weights  is the array of the weights used to combine the simulated reference spectra into the simulated sample spectra

        if X_data is not None:
            if "MC" in preproc:
                self.centring = np.mean(X_data, 1)  # calculate the mean of the data
                X_data = (
                    X_data.transpose() - self.centring
                ).transpose()  # mean centre data
            elif "MdnC" in preproc:
                self.centring = np.median(X_data, 1)  # calculate the mean of the data
                X_data = (
                    X_data.transpose() - self.centring
                ).transpose()  # mean centre data
            else:
                self.mean = 0

            if "UV" in preproc:
                self.scale = X_data.std(1)
                X_data = (X_data.transpose() / self.scale).transpose()
            elif "MinMax" in preproc:
                self.scale = X_data.max(1) - X_data.min(1)
            elif "MaxAbs" in preproc:
                self.scale = abs(X_data).max(1)
            else:
                self.scale = 1

            self.X = X_data
            self.N_Vars, self.N_Obs = (
                X_data.shape
            )  # calculate v (N_Vars) and o (N_Obs), the number of variables and observations in the data matrX_data
        else:
            print("No data provided")
            return

        # maximum_number_PCs is the maximum number of principle components to calculate
        if maximum_number_PCs is not None:
            #            print(type(maximum_number_PCs))
            self.N_PC = min(
                X_data.shape[0], X_data.shape[1], maximum_number_PCs
            )  # ensure max_i is achievable (minimum of v,o and max_i)

        else:
            self.N_PC = min(X_data.shape[0], X_data.shape[1])

        self.pCon = np.empty([self.N_Vars, self.N_PC])
        self.pCon[:] = np.nan
        self.nCon = np.copy(self.pCon)
        self.cCon = np.copy(self.pCon)
        self.mCon = np.copy(self.pCon)
        self.pConS = np.copy(self.pCon)
        self.nConS = np.copy(self.pCon)
        self.cConS = np.copy(self.pCon)
        self.mConS = np.copy(self.pCon)
        self.optSF = np.empty(self.N_PC)
        self.optSF[:] = np.nan
        
        self.fig_Size = [8,8]
        self.fig_Resolution = 300
        self.fig_Format = 'png'
        self.fig_k = range( 0 , self.N_Obs , np.ceil(self.N_Obs/10).astype(int) )
        self.fig_i = range( 0 , np.min((self.N_PC , 5 ) ) )
        self.fig_Y_Label = ''
        self.fig_X_Label = ''
        self.fig_Show_Labels =  False
        self.fig_Show_Values =  False
        self.fig_Text_Size =  12
        self.fig_Project = 'Graphical PCA Demo'
        
        # maximum_iterations_PCs is the maximum number of iterations to perform for each PC
        if maximum_iterations_PCs is not None:
            self.Max_It = maximum_iterations_PCs

            # iteration_tolerance is the tolerance for decieding if subsequent iterations are significantly reducing the residual variance
        if iteration_tolerance is not None:
            #           print('iteration_tolerance not Empty')
            self.Tol = iteration_tolerance
        else:
            print("iteration_tolerance Empty")

        if pixel_axis is not None:
            self.pixel_axis = pixel_axis
        else:
            self.pixel_axis = list(range(self.N_Vars))

        self.REigenvector = np.ndarray(
            (self.N_PC, self.N_Vars)
        )  # initialise array for right eigenvector (loadings/principal components/latent factors for column oriented matrices, note is transposed wrt original data)
        self.LEigenvector = np.ndarray(
            (self.N_Obs, self.N_PC)
        )  # initialise array for left eigenvector (scores for column oriented matrices)
        self.r = (
            []
        )  # initialise list for residuals for each PC (0 will be initial data, so indexing will equal the PC number)
        self.pc = (
            []
        )  # initialise list for iterations on the right eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.w = (
            []
        )  # initialise list for iterations on the left eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.rE = np.ndarray(
            (self.N_PC + 1, self.N_Vars)
        )  # residual error at each pixel for each PC (0 will be initial data, so indexing will equal the PC number)
        self.spectral_weights = spectral_weights
        self.min_spectral_values = min_spectral_values
        return
    
    def figure_Settings( 
        self, 
        size=None, 
        res=None, 
        frmt=None, 
        k=None, 
        i=None, 
        Ylab=None, 
        Xlab=None, 
        Show_Labels=None, 
        Show_Values=None, 
        TxtSz=None,
        Project=None
    ):
        # NOTE that the user is expected to update all settings in one call, otherwise excluded settings will revert to default
        if self.X is None:
                print('Data must be loaded into object in order to define plot parameters')
                return

        if size is not None:
            if np.ndim(size) == 1 and np.shape(size)[0] == 2:
                self.fig_Size = size
            else:
                print('Size should be a two value vector, reverting to default size of 8 x 8 cm')
                self.fig_Size = [ 8 , 8 ]
        else:
            self.fig_Size = [ 8 , 8 ]

        if res is not None:
            if np.ndim(res) == 1 and np.shape(res)[0] == 1:
                self.fig_Resolution = res
            else:
                print('Resolution should be a scalar, reverting to default resolution of 300 dpi')
                self.fig_Resolution = 300
        else:
            self.fig_Resolution = 300

        # dynamically get list of supported formats, based on accepted answer at https://stackoverflow.com/questions/7608066/in-matplotlib-is-there-a-way-to-know-the-list-of-available-output-format
        # class has imported matplotlib so shouldn't need imported as in the answer
        fig = plt.figure()
        availFrmt = fig.canvas.get_supported_filetypes()
        if frmt is not None:
            if frmt in availFrmt:
                self.fig_Format = frmt
            else:
                print('Format string not recognised, reverting to default of png')
                self.fig_Format = 'png'
        else:
            self.fig_Format = 'png'

        default_K = range( 0 , self.N_Obs , np.ceil(self.N_Obs/10).astype(int))
        if k is not None:
            if np.shape(k)[0]<3:
                print('a minimum of 3 samples must be plotted to observe variation. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.shape(k)[0]>10:
                print('a maximum of 10 samples permitted to prevent overcrowding. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.min(k)<0:
                print('Cannot start indexing before 0. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.max(k)>self.N_Obs:
                print('Cannot index beyond number of observations. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            else:
                self.fig_k = k           
        else:
            self.fig_k = default_K

        default_i = range( 0 , np.min((self.N_PC , 4, np.max(self.fig_k)-1) ) ) # N_PCs constrained to be less than k in local_NIPALS init
        if i is not None:
            if np.shape(i)[0] < 2:
                print('a minimum of 2 pcs must be plotted to observe variation. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.max(i) > (np.max(self.fig_k)-1):                     
                print('a maximum of 9 pcs permitted to prevent overcrowding. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.min(i)<0:
                print('Cannot start indexing before 0. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.max(i)>self.N_PC:
                print('Cannot index beyond number of PCs. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            else:
                self.fig_i = i            
        else:
            self.fig_i = default_i


        if Ylab is not None:
            if isinstance(Ylab, str):
                if len(Ylab)<33:
                    self.fig_Y_Label = Ylab
                else:
                    print('Y label too long, truncating to 32 characters')
                    self.fig_Y_Label = Ylab[:31]
            else:
                print('Y label is not a string, no Y label will be displayed')
                self.fig_Y_Label = ''
        else:
            self.fig_Y_Label = ''

        if Xlab is not None:
            if isinstance(Xlab, str):
                if len(Xlab)<33:
                    self.fig_X_Label = Xlab
                else:
                    print('X label too long, truncating to 32 characters')
                    self.fig_X_Label = Xlab[:31]
            else:
                print('X label is not a string, no X label will be displayed')
                self.fig_X_Label = ''
        else:
            self.fig_X_Label = ''

        if Show_Labels is not None:
            if Show_Labels and (len(self.Y_Label)>0 or len(self.X_Label)>0):
                self.fig_Show_Labels =  True
            else:
                self.fig_Show_Labels =  False        
        else:
            self.fig_Show_Labels =  False

        if Show_Values is not None:
            if Show_Values:
                self.fig_Show_Values =  True
            else:
                self.fig_Show_Values =  False        
        else:
            self.fig_Show_Values =  False

        if  TxtSz is not None:
            if TxtSz<8:
                print('Specified text size is below limit of 8, defaulting to 8')
                self.fig_Text_Size =  8
            elif TxtSz>18:
                print('Specified text size is above limit of 18, defaulting to 18')
                self.fig_Text_Size =  18
            else:
                self.fig_Text_Size =  TxtSz
        else:
            self.fig_Text_Size =  12
        
        if Project is not None:
            if isinstance(Project, str):
                if len(Project)<25:
                    self.fig_Project = Project
                else:
                    print('Project code too long, truncating to 25 characters')
                    self.fig_Project = Project[:24]
            else:
                print('Project code is not a string, default code "Graphical PCA Demo" applied')
                self.fig_Project = 'Graphical PCA Demo'
        else:
            self.fig_Project = 'Graphical PCA Demo'
           
            
    def calc_PCA(self):
        #        print('initialising NIPALS algorithm')
        self.r.append(self.X)  # initialise the residual_i as the raw input data
        self.rE[0, :] = np.sum(
            self.r[0] ** 2, 1
        )  # calculate total variance (initial residual variance)

        for iPC in range(self.N_PC):  # for i = 0,1,... max_i
            pc = np.ndarray((self.N_Vars, self.Max_It))
            w = np.ndarray((self.Max_It, self.N_Obs))
            jIt = 0  # iteration counter initialised
            pc[:, jIt] = (
                np.sum(self.r[iPC] ** 2, 1) ** 0.5
            )  # pc_i,j = norm of sqrt(sum(residual_i^2))
            pc[:, jIt] = (
                pc[:, jIt] / sum(pc[:, jIt] ** 2) ** 0.5
            )  # convert to unit vector for initial right eigenvector guess

            itChange = sum(
                abs(pc[:, 0])
            )  # calculate the variance in initialisation iteration, itChange = |pc_i,j|

            while True:
                if (
                    jIt < self.Max_It - 1 and itChange > self.Tol
                ):  # Max_It-1 since 0 is the first index: for j = 1,2,... max_j, while itChange<tol
                    #                    print(str(jIt)+' of '+str(self.Max_It)+' iterations')
                    #                    print(str(diff)+' compared to tolerance of ',str(self.Tol))
                    w[jIt, :] = (
                        self.r[iPC].T @ pc[:, jIt]
                    )  # calculate LEigenvectors from REigenvectors: w_i,j = outer product( residual_i , pc_i,j )
                    jIt += 1  # reset iteration counter, j, to 0
                    pc[:, jIt] = np.inner(
                        w[jIt - 1, :].T, self.r[iPC]
                    )  # update estimate of REigenvectors using LEigenvectors resulting from previous estimate: pc_i,j = inner product( w'_i,j-1 , residual_i)
                    ml = sum(pc[:, jIt] ** 2) ** 0.5  # norm of REigenvectors
                    pc[:, jIt] = (
                        pc[:, jIt] / ml
                    )  # convert REigenvectors to unit vectors:  pc_i,j = pc_i,j/|pc_i,j|
                    itChange = sum(
                        abs(pc[:, jIt] - pc[:, jIt - 1])
                    )  # total difference between iterations: itChange = sum(|pc_i,j - pc_i,j-1|)

                else:
                    break
                # endfor

            self.pc.append(
                pc[:, 0 : jIt - 1]
            )  # truncate iteration vectors to those calculated prior to final iteration
            self.REigenvector[iPC, :] = pc[
                :, jIt
            ]  # store optimised REigenvector: rEV_i = pc_i,j

            self.w.append(
                w[0 : jIt - 1, :]
            )  # truncate iteration vectors to those calculated
            self.LEigenvector[:, iPC] = (
                self.r[iPC].T @ pc[:, jIt]
            )  # store LEigenvectors, AKA scores: lEV_i = outer product( residual_i , pc_i,j )
            self.r.append(
                self.r[iPC]
                - np.outer(self.LEigenvector[:, iPC].T, self.REigenvector[iPC, :].T).T
            )  # update residual:     residual_i+1 = residual_i - outer product( lEV_i, rEV_i )
            self.rE[iPC + 1, :] = np.sum(
                self.r[iPC + 1] ** 2, 1
            )  # calculate residual variance

    def calc_Constituents(self, iPC):
        # calculate the constituents that comprise the PC, splitting them into 3 parts:
        # Positive score weighted summed spectra - positive contributors to the PC
        # -Negative score weighted summed spectra - negative contributors to the PC
        # Sparse reconstruction ignoring current PC - common contributors to both the positive and negative constituent
        # Note that this implmenation is not generalisable, it depends on knowledge that is not known in real world datasets,
        # namely what the global minimum intensity at every pixel is. This means it is only applicable for simulated datasets.
        if self.mean != 0:
            print(
                "Warning: using subtracted spectra, such as mean or median centred data, will result in consitutents that are still"
                + " subtraction spectra - recontructing these to positive signals requires addition of the mean or median"
            )
        #        print('extracting contributing LEigenvector features')
        posSc = self.LEigenvector[:, iPC] > 0

        self.nCon[:, iPC] = -np.sum(
            self.LEigenvector[posSc == False, iPC] * self.X[:, posSc == False], axis=1
        )
        self.pCon[:, iPC] = np.sum(
            self.LEigenvector[posSc, iPC] * self.X[:, posSc], axis=1
        )
        self.cCon[:, iPC] = np.inner(
            np.inner(
                np.mean([self.nCon[:, iPC], self.pCon[:, iPC]], axis=0),
                self.REigenvector[range(iPC), :],
            ),
            self.REigenvector[range(iPC), :].transpose(),
        )
        self.mCon[:, iPC] = np.sum(
            self.LEigenvector[posSc, iPC] * self.min_spectral_values[:, posSc], axis=1
        )  # minimum score vector

        tSF = np.ones(11)
        res = np.ones(11)
        for X_data in range(
            0, 9
        ):  # search across 10x required precision of last minimum
            tSF[X_data] = (X_data + 1) / 10
            nConS = (
                self.nCon[:, iPC] - self.cCon[:, iPC] * tSF[X_data]
            )  # negative constituent corrected for common signal
            pConS = (
                self.pCon[:, iPC] - self.cCon[:, iPC] * tSF[X_data]
            )  # positive constituent corrected for common signal
            mConS = self.mCon[:, iPC] * (1 - tSF[X_data])
            cConS = self.cCon[:, iPC] * (1 - tSF[X_data])

            res[X_data] = np.min([nConS - mConS, pConS - mConS]) / np.max(cConS)
        res[res < 0] = np.max(
            res
        )  # set all negative residuals to max so as to bias towards undersubtraction
        optSF = tSF[np.nonzero(np.abs(res) == np.min(np.abs(res)))]

        for iPrec in range(2, 10):  # define level of precsision required
            tSF = np.ones(19)
            res = np.ones(19)
            for X_data in range(
                -9, 10
            ):  # serach across 10x required precision of last minimum
                tSF[X_data + 9] = optSF + X_data / 10 ** iPrec
                nConS = (
                    self.nCon[:, iPC] - self.cCon[:, iPC] * tSF[X_data + 9]
                )  # - constituent corrected for common signal
                pConS = (
                    self.pCon[:, iPC] - self.cCon[:, iPC] * tSF[X_data + 9]
                )  # + constituent corrected for common signal
                cConS = self.cCon[:, iPC] * (1 - tSF[X_data + 9])
                mConS = self.mCon[:, iPC] * (1 - tSF[X_data + 9])

                res[X_data + 9] = np.min([nConS - mConS, pConS - mConS]) / np.max(cConS)
            res[res < 0] = np.max(
                res
            )  # set all negative residuals to max so as to bias towards undersubtraction
            optSF = tSF[np.nonzero(np.abs(res) == np.min(np.abs(res)))]
        self.optSF[iPC] = optSF[0]
        self.nConS[:, iPC] = (
            self.nCon[:, iPC] - self.cCon[:, iPC] * optSF
        )  # - constituent corrected for common signal
        self.pConS[:, iPC] = (
            self.pCon[:, iPC] - self.cCon[:, iPC] * optSF
        )  # + constituent corrected for common signal
        self.cConS[:, iPC] = self.cCon[:, iPC] * (1 - optSF)
        self.mConS[:, iPC] = self.mCon[:, iPC] * (1 - optSF)

        ## need to work out how to handle min_spectral_values

        # print('extracted positive and negative LEigenvector features')
                    
    def figure_DSLT(self):
        
        figDSLT, axDSLT = plt.subplots(1, 3, figsize=(8, 8))
        axDSLT[0] = plt.subplot2grid((5, 20), (0, 0), colspan=8, rowspan=5)
        axDSLT[1] = plt.subplot2grid((5, 20), (0, 9), colspan=2, rowspan=5)
        axDSLT[2] = plt.subplot2grid((5, 20), (1, 12), colspan=8, rowspan=3)
        
        data4plot = self.X[:,self.fig_k]
        data_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(data4plot,axis=0))/2)
        data4plot = data4plot + data_spacing
        data0lines = np.tile([data_spacing],(2,1))
        
        REigenvectors4plot = self.REigenvector[self.fig_i,:].transpose()
        REig_spacing = -(np.arange(np.shape(self.fig_i)[0]))*(np.mean(np.max(REigenvectors4plot,axis=1))*4)
        REigenvectors4plot = REigenvectors4plot + REig_spacing
        REig0lines = np.tile([REig_spacing],(2,1))
        
        LEigenvectors4plot = self.LEigenvector[self.fig_k,:]
        LEigenvectors4plot = LEigenvectors4plot[:,self.fig_i]
        LEig_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(LEigenvectors4plot,axis=1))*1)
        LEigenvectors4plot = (LEigenvectors4plot.transpose() + LEig_spacing).transpose()
        LEig0lines = np.tile([LEig_spacing],(2,1))

      
        axDSLT[0].plot(self.pixel_axis, data4plot)
        axDSLT[0].plot(self.pixel_axis[[0,-1]],data0lines)
        axDSLT[1].plot(LEigenvectors4plot.transpose(), ".")
        axDSLT[1].plot([0,np.shape(self.fig_i)[0]],LEig0lines, "-.")
        axDSLT[2].plot(self.pixel_axis,REigenvectors4plot)
        axDSLT[2].plot(self.pixel_axis[[0,-1]],REig0lines,'-.')
        #,transform=transforms.Affine2D().rotate_deg(90) + plt.gca().transData,)
        # originally printed in 90o rotation, but this applies to column vectors - D is displayed as a row vector, so LT should also be row. the transpose of S .LT returns D as a row vector. By cpnvention PCA done on column vectors. Need to handle this so as not to cause confusion but to retain consistency with math
        for iC in range(np.shape(self.fig_i)[0]):
            axDSLT[2].lines[iC].set_color(str(0 + iC / 5)) #shade loadings
            axDSLT[2].lines[iC+np.shape(self.fig_i)[0]].set_color(str(0 + iC / 5)) #shade zero lines


        axDSLT[0].annotate(
            "$D_{-\mu}$",
            xy=(0.25, 0.95),
            xytext=(0.25, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axDSLT[0].annotate(
            "$k=1$",
            xy=(0.08, 0.07),
            xytext=(0.08, 0.2),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[0].annotate(
            "$k=n$",
            xy=(0.08, 0.06),
            xytext=(0.08, 0.06),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[0].annotate(
            "$j=1$",
            xy=(0.2, 0.04),
            xytext=(0.1, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )
        axDSLT[0].annotate(
            "$j=p$",
            xy=(0.2, 0.04),
            xytext=(0.2, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
            va="center",
        )
        axDSLT[0].annotate(
            "$nxp$",
            xy=(0.15, 0.12),
            xytext=(0.15, 0.12),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )

        axDSLT[1].annotate(
            "=",
            xy=(0.45, 0.5),
            xytext=(0.45, 0.5),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*1.5,
            horizontalalignment="center",
        )

        axDSLT[1].annotate(
            "$S$",
            xy=(0.1, 0.95),
            xytext=(0.52, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axDSLT[1].annotate(
            "$k=1$",
            xy=(0.45, 0.07),
            xytext=(0.45, 0.2),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[1].annotate(
            "$k=n$",
            xy=(0.45, 0.06),
            xytext=(0.45, 0.06),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[1].annotate(
            "$i=1$",
            xy=(0.57, 0.04),
            xytext=(0.47, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )
        axDSLT[1].annotate(
            "$i=d$",
            xy=(0.57, 0.04),
            xytext=(0.57, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
            va="center",
        )

        axDSLT[1].annotate(
            "$nxd$",
            xy=(0.52, 0.12),
            xytext=(0.52, 0.12),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )
        axDSLT[2].annotate(
            r"$\cdot$",
            xy=(0.6, 0.5),
            xytext=(0.6, 0.5),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*3,
            horizontalalignment="center",
        )

        axDSLT[2].annotate(
            r"$L{^\top}$",
            xy=(0.85, 0.95),
            xytext=(0.8, 0.95),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[2].annotate(
            "$i=1$",
            xy=(0.67, 0.07),
            xytext=(0.67, 0.2),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[2].annotate(
            "$i=d$",
            xy=(0.67, 0.06),
            xytext=(0.67, 0.06),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[2].annotate(
            "$j=1$",
            xy=(0.82, 0.04),
            xytext=(0.69, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )
        axDSLT[2].annotate(
            "$dxp$",
            xy=(0.75, 0.12),
            xytext=(0.75, 0.12),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )
        axDSLT[2].annotate(
            "$j=p$",
            xy=(0.83, 0.04),
            xytext=(0.83, 0.04),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
            va="center",
        )

        if not self.fig_Show_Values: 
            for iax in range(len(axDSLT)):
                axDSLT[iax].axis("off")
            
        if self.fig_Show_Labels:
            axDSLT[0].set_ylabel(self.fig_Y_Label)
            axDSLT[0].set_xlabel(self.fig_X_Label)
            axDSLT[1].set_ylabel('Score / Arbitrary')
            axDSLT[1].set_xlabel('Sample #')
            axDSLT[2].set_ylabel('Weights / Arbitrary')
            axDSLT[2].set_xlabel(self.fig_X_Label)
            
        figDSLT.savefig(str(images_folder) + "\\"+self.fig_Project + " DSLTmainEqn.png", dpi=self.fig_Resolution)
        plt.close()
    
    def figure_lpniCommonSignalScalingFactors(self, nPC, xview):
        ###################       START lpniCommonSignalScalingFactors       #######################
        # FIGURE of the scaling factor calculated for subtracting the common signal from the positive
        # and negative constituents of a PC

        pixel_axis = self.pixel_axis

        for iPC in range(1, nPC):
            # generate subtraction figures for positive vs negative score weighted sums.
            self.calc_Constituents(iPC)
            self.figure_lpniLEigenvectorEqn(iPC, pixel_axis)
            self.figure_lpniCommonSignal(iPC, pixel_axis)

        figlpniS, axlpniS = plt.subplots(
            1, 6, figsize=(8, 8)
        )  # Extra columns to match spacing in
        for iPC in range(1, nPC):
            iFig = (
                iPC % 6 - 1
            )  # modulo - determine where within block the current PC is
            if iFig == -1:
                iFig = 5  # if no remainder then it is the last in the cycle
            axlpniS[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniS[iFig].plot(self.pixel_axis[[xview[0], xview[-0]]], [0, 0], "--")
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.nConS[xview, iPC], "b", linewidth=1
            )
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.pConS[xview, iPC], "y", linewidth=1
            )
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.cConS[xview, iPC], "g", linewidth=0.5
            )  # *smsf[iPC]
            txtpos = [
                np.mean(axlpniS[iFig].get_xlim()),
                axlpniS[iFig].get_ylim()[1] * 0.9,
            ]
            axlpniS[iFig].annotate(
                "PC " + str(iPC + 1),
                xy=(txtpos),
                xytext=(txtpos),
                textcoords="data",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
            )

            if iFig == 5:
                for iax in range(np.shape(axlpniS)[0]):
                    axlpniS[iax].axis("off")

                image_name = f"lpniSubLEigenvectorEqn_{str(iPC - 4)}_{str(iPC + 1)}.png"
                figlpniS.savefig(images_folder / image_name, dpi=300)
                plt.close()
                figlpniS, axlpniS = plt.subplots(
                    1, 6, figsize=(8, 8)
                )  # Extra columns to match spacing in

        plt.figure(figsize=(8, 8))
        figsmsf = plt.plot(range(2, np.shape(self.optSF)[0] + 1), self.optSF[1:], ".")
        drp = np.add(np.nonzero((self.optSF[2:] - self.optSF[1:-1]) < 0), 2)
        if np.size(drp) != 0:
            plt.plot(drp + 1, self.optSF[drp][0], "or")
            plt.plot([2, nPC], [self.optSF[drp][0], self.optSF[drp][0]], "--")
        plt.savefig(images_folder / "lpniCommonSignalScalingFactors.png", dpi=300)
        plt.close()

        # copy scalingAdjustment.py into its own cell after running this main cell in Jupyter then you
        # can manually adjust the scaling factors for each PC to determine what is the most appropriate method

        ###### Plot positive, negative score and common  signals without any  common signal subtraction ######
        figlpniU, axlpniU = plt.subplots(
            1, 6, figsize=(8, 8)
        )  # Extra columns to match spacing
        for iFig in range(6):
            axlpniU[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniU[iFig].plot(self.pixel_axis[[xview[0], xview[-0]]], [0, 0], "--")
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.nCon[xview, iFig + 1], "b", linewidth=1
            )
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.pCon[xview, iFig + 1], "y", linewidth=1
            )
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.cCon[xview, iFig + 1], "g", linewidth=0.5
            )  # *smsf[iPC]
            axlpniU[iFig].annotate(
                "PC " + str(iFig + 2),
                xy=(
                    np.mean(axlpniU[iFig].get_xlim()),
                    axlpniU[iFig].get_ylim()[1] * 0.9,
                ),
                xytext=(
                    np.mean(axlpniU[iFig].get_xlim()),
                    axlpniU[iFig].get_ylim()[1] * 0.9,
                ),
                textcoords="data",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
            )
        for iax in range(np.shape(axlpniU)[0]):
            axlpniU[iax].axis("off")
        figlpniU.savefig(images_folder / "lpniUnSubLEigenvectorEqn_2_4.png", dpi=300)
        plt.close()
        ###################         END lpniCommonSignalScalingFactors           #######################

    def figure_lpniLEigenvectorEqn(self, iPC, pixel_axis):
        # this class function prints out tiff images comparing the score magnitude weighted summed spectra for
        # positive and negative score spectra. The class must have already calculated the positive, negative
        # and common consitituents
        if ~np.isnan(self.cCon[0, iPC]):

            figlpni, axlpni = plt.subplots(1, 6, figsize=(8, 8))
            axlpni[0] = plt.subplot2grid((1, 21), (0, 0), colspan=1)
            axlpni[1] = plt.subplot2grid((1, 21), (0, 1), colspan=6)
            axlpni[2] = plt.subplot2grid((1, 21), (0, 7), colspan=1)
            axlpni[3] = plt.subplot2grid((1, 21), (0, 8), colspan=6)
            axlpni[4] = plt.subplot2grid((1, 21), (0, 14), colspan=1)
            axlpni[5] = plt.subplot2grid((1, 21), (0, 15), colspan=6)
            posSc = self.LEigenvector[:, iPC] > 0  # skip PC1 as not subtraction

            axlpni[0].plot([-10, 10], np.tile(0, (2, 1)), "k")
            axlpni[0].plot(np.tile(0, sum(posSc)), self.LEigenvector[posSc, iPC], ".y")
            axlpni[0].plot(
                np.tile(0, sum(posSc == False)),
                self.LEigenvector[posSc == False, iPC],
                ".b",
            )

            axlpni[1].plot(pixel_axis, self.X[:, posSc], "y")
            axlpni[1].plot(pixel_axis, self.X[:, posSc == False], "--b", lw=0.1)
            axlpni[1].annotate(
                "$s_i$",
                xy=(0.1, 0.9),
                xytext=(pixel_axis[0] - 110, 0.9),
                textcoords="data",
                fontsize=self.fig_Text_Size*1.5,
                horizontalalignment="left",
            )

            axlpni[3].plot(pixel_axis, self.nCon[:, iPC], "b")
            axlpni[3].plot(pixel_axis, self.pCon[:, iPC], "y")

            pnCon = self.pCon[:, iPC] - self.nCon[:, iPC]
            pnCon = pnCon / np.sum(pnCon ** 2) ** 0.5

            axlpni[3].annotate(
                "$s_i*d_i$",
                xy=(0.1, 0.9),
                xytext=(0.6, 0.9),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[5].plot(self.REigenvector[iPC, :], "m")
            axlpni[5].plot(pnCon, "c")
            axlpni[5].plot(self.REigenvector[iPC, :] - pnCon, "--k")

            ylim = np.max(pnCon)
            axlpni[5].annotate(
                "$L$",
                xy=(0.1, 0.9),
                xytext=(0, 0.9),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="m",
            )
            axlpni[5].annotate(
                "$p-n$",
                xy=(0.1, 0.9),
                xytext=(0, 0.85),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="c",
            )
            axlpni[5].annotate(
                "$L-(p-n)$",
                xy=(0.1, 0.9),
                xytext=(0, 0.8),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="k",
            )
            totdiff = "{:.0f}".format(
                np.log10(np.mean(np.abs(self.REigenvector[iPC, :] - pnCon)))
            )
            axlpni[5].annotate(
                "$\Delta_{L,p-n}=10^{" + totdiff + "}$",
                xy=(0.1, 0.9),
                xytext=(0.1, 0.77),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="k",
            )

            axlpni[2].annotate(
                "$\Sigma(|s_i^+|*d_i^+)$",
                xy=(0, 0.5),
                xytext=(0.5, 0.55),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[2].annotate(
                "$\Sigma(|s_i^-|*d_i^-)$",
                xy=(0, 0.5),
                xytext=(0.5, 0.42),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[2].annotate(
                "",
                xy=(1, 0.5),
                xytext=(0, 0.5),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )
            axlpni[4].annotate(
                "$sd_i^+-sd_i^-$",
                xy=(0, 0.5),
                xytext=(0.5, 0.55),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[4].annotate(
                "",
                xy=(1, 0.5),
                xytext=(0, 0.5),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )

            for iax in range(np.shape(axlpni)[0]):
                axlpni[iax].axis("off")
            filename = f"lpniLEigenvectorEqn_{str(iPC + 1)}.png"
            figlpni.savefig(images_folder / filename, dpi=300)
            plt.close()
        else:
            print(
                "Common, Positive and Negative Consituents must be calculated first using calcCons"
            )

    def figure_lpniCommonSignal(self, iPC, pixel_axis):
        # this class function prints out tiff images comparing the score magnitude weighted summed spectra for
        # positive and negative score spectra corrected for the common consitituents, compared with the common
        # consituents itself and the scaled global minimum

        plt.plot(pixel_axis, self.nConS[:, iPC], "b")
        plt.plot(pixel_axis, self.pConS[:, iPC], "y")
        plt.plot(pixel_axis, self.cConS[:, iPC], "g")
        plt.plot(pixel_axis, self.min_spectral_values, "c")
        plt.title("PC" + str(iPC) + " Scale Factor:" + str(self.optSF[iPC]))
        plt.legend(
            ("-ve Constituent", "+ve Constituent", "Common Signal", "Global Minimum")
        )
        filename = f"lpniDeterminingCommonSignalScalingFactorsPC_{str(iPC + 1)}.png"
        plt.savefig(images_folder / filename, dpi=300)
        plt.close()

