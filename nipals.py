import numpy as np
import matplotlib.pyplot as plt

class NIPALS:
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many 
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned 
    # along the columns and each row corresponds to different variables
    
    #comments include references to relevant lines in the pseudocode listed in the paper
    
    def __init__( self , iX , iNPC , iMxIt=20 , iTol=0.00001 , preproc='None' , xcal='None' , refWts = None , minSpec = None ):
        # requires a data matrix and optional additional settings
        # iX      X, main data matrix as specified in pseudocode
        # iNPC    is max_i, desired maximum number of PCs
        # iMxIt   is max_j, maximum iterations on each PC
        # iTol    is tol, acceptable tolerance for difference between j and j-1 estimates of the PCs
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
        # xcal    is a vector of values for each pixel in the x-axis or a tuple specifying a fixed interval spacing 
        #         (2 values for unit spacing, 3 values for non-unit spacing)
        # refWts  is the array of the weights used to combine the simulated reference spectra into the simulated sample spectra
        
        if iX is not None:
            if 'MC' in preproc:
                self.centring  = np.mean(iX,1) # calculate the mean of the data
                iX = (iX.transpose()-self.centring).transpose() # mean centre data
            elif 'MdnC' in preproc:
                self.centring  = np.median(iX,1) # calculate the mean of the data
                iX = (iX.transpose()-self.centring).transpose() # mean centre data                
            else:
                self.mean = 0
                    
            if 'UV' in preproc:
                self.scale = iX.std(1)
                iX = (iX.transpose()/self.scale).transpose()
            elif 'MinMax' in preproc:
                self.scale = iX.max(1) - iX.min(1)
            elif 'MaxAbs' in preproc:
                self.scale = abs(iX).max(1)
            else:
                self.scale = 1

            self.X = iX
            self.N_Vars, self.N_Obs = iX.shape #calculate v (N_Vars) and o (N_Obs), the number of variables and observations in the data matrix
        else:
            print('No data provided')
            return
            
        #iNPC is the maximum number of principle components to calculate
        if iNPC is not None:
#            print(type(iNPC))
            self.N_PC = min(iX.shape[0],iX.shape[1],iNPC) #ensure max_i is achievable (minimum of v,o and max_i)

        else:
            self.N_PC = min(iX.shape[0],iX.shape[1])

        self.pCon = np.empty([ self.N_Vars , self.N_PC ])
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
        
        1# iMxIt is the maximum number of iterations to perform for each PC
        if iMxIt is not None:
            self.Max_It = iMxIt 
            
        #iTol is the tolerance for decieding if subsequent iterations are significantly reducing the residual variance
        if iTol is not None:
 #           print('iTol not Empty')
            self.Tol = iTol 
        else:
            print('iTol Empty')
        
        if xcal is not None:
            self.xcal = xcal
        else:
            self.xcal = list(range(self.N_Vars))
        
        self.REigenvector = np.ndarray((self.N_PC, self.N_Vars))# initialise array for right eigenvector (loadings/principal components/latent factors for column oriented matrices, note is transposed wrt original data)
        self.LEigenvector = np.ndarray((self.N_Obs, self.N_PC))#initialise array for left eigenvector (scores for column oriented matrices)
        self.r = [] # initialise list for residuals for each PC (0 will be initial data, so indexing will equal the PC number)
        self.pc = [] # initialise list for iterations on the right eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.w = [] # initialise list for iterations on the left eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.rE = np.ndarray((self.N_PC+1, self.N_Vars)) #residual error at each pixel for each PC (0 will be initial data, so indexing will equal the PC number)
        self.refWts = refWts
        self.minSpec = minSpec
        return

       
    def calc_PCA( self ):
#        print('initialising NIPALS algorithm')
        self.r.append(self.X)  #initialise the residual_i as the raw input data
        self.rE[0,:] = np.sum(self.r[0]**2,1) # calculate total variance (initial residual variance)

        for iPC in range(self.N_PC): #for i = 0,1,... max_i
            pc = np.ndarray((self.N_Vars,self.Max_It))
            w = np.ndarray((self.Max_It, self.N_Obs))
            jIt = 0 # iteration counter initialised
            pc[:,jIt] = np.sum(self.r[iPC]**2,1)**0.5 #pc_i,j = norm of sqrt(sum(residual_i^2))
            pc[:,jIt] = pc[:,jIt]/sum(pc[:,jIt]**2)**0.5 # convert to unit vector for initial right eigenvector guess

            itChange = sum(abs(pc[:,0])) # calculate the variance in initialisation iteration, itChange = |pc_i,j|

            while True:
                if jIt<self.Max_It-1 and itChange>self.Tol: #Max_It-1 since 0 is the first index: for j = 1,2,... max_j, while itChange<tol
#                    print(str(jIt)+' of '+str(self.Max_It)+' iterations')
#                    print(str(diff)+' compared to tolerance of ',str(self.Tol))
                    w[jIt,:] = self.r[iPC].T@pc[:,jIt] # calculate LEigenvectors from REigenvectors: w_i,j = outer product( residual_i , pc_i,j ) 
                    jIt += 1 # reset iteration counter, j, to 0
                    pc[:,jIt] = np.inner(w[jIt-1,:].T,self.r[iPC]) #update estimate of REigenvectors using LEigenvectors resulting from previous estimate: pc_i,j = inner product( w'_i,j-1 , residual_i) 
                    ml = sum(pc[:,jIt]**2)**0.5 # norm of REigenvectors
                    pc[:,jIt] = pc[:,jIt]/ml # convert REigenvectors to unit vectors:  pc_i,j = pc_i,j/|pc_i,j|
                    itChange = sum(abs(pc[:,jIt]-pc[:,jIt-1])) #total difference between iterations: itChange = sum(|pc_i,j - pc_i,j-1|)

                else:
                    break
                # endfor

            self.pc.append(pc[:,0:jIt-1]) #truncate iteration vectors to those calculated prior to final iteration
            self.REigenvector[iPC,:] = pc[:,jIt] #store optimised REigenvector: rEV_i = pc_i,j
            
            self.w.append(w[0:jIt-1,:]) #truncate iteration vectors to those calculated
            self.LEigenvector[:,iPC] = self.r[iPC].T@pc[:,jIt] # store LEigenvectors, AKA scores: lEV_i = outer product( residual_i , pc_i,j ) 
            self.r.append(self.r[iPC] -  np.outer(self.LEigenvector[:,iPC].T,self.REigenvector[iPC,:].T).T) # update residual:     residual_i+1 = residual_i - outer product( lEV_i, rEV_i )
            self.rE[iPC+1,:] = np.sum(self.r[iPC+1]**2,1) # calculate residual variance

    
    def calc_Constituents( self , iPC ):
        # calculate the constituents that comprise the PC, splitting them into 3 parts:
            # Positive score weighted summed spectra - positive contributors to the PC
            # -Negative score weighted summed spectra - negative contributors to the PC
            # Sparse reconstruction ignoring current PC - common contributors to both the positive and negative constituent
        # Note that this implmenation is not generalisable, it depends on knowledge that is not known in real world datasets, 
        # namely what the global minimum intensity at every pixel is. This means it is only applicable for simulated datasets.
        if self.mean != 0:
            print('Warning: using subtracted spectra, such as mean or median centred data, will result in consitutents that are still'
                  + ' subtraction spectra - recontructing these to positive signals requires addition of the mean or median')
#        print('extracting contributing LEigenvector features')
        posSc = self.LEigenvector[:,iPC]>0

        self.nCon[:,iPC] = -np.sum( self.LEigenvector[posSc==False,iPC] * self.X[:,posSc==False] , axis=1 )
        self.pCon[:,iPC] = np.sum( self.LEigenvector[posSc,iPC] * self.X[:,posSc] , axis=1 )
        self.cCon[:,iPC] = np.inner( np.inner( np.mean( [self.nCon[:,iPC],self.pCon[:,iPC]] , axis=0 ) , 
                                              self.REigenvector[ range(iPC) , : ]) , self.REigenvector[ range(iPC) , : ].transpose())
        self.mCon[:,iPC] = np.sum( self.LEigenvector[posSc,iPC] * self.minSpec[:,posSc] , axis=1 ) #minimum score vector

        tSF = np.ones(11)
        res = np.ones(11)
        for ix in range(0,9): #search across 10x required precision of last minimum
            tSF[ix] = (ix+1)/10
            nConS = self.nCon[:,iPC]-self.cCon[:,iPC]*tSF[ix] #negative constituent corrected for common signal
            pConS = self.pCon[:,iPC]-self.cCon[:,iPC]*tSF[ix] #positive constituent corrected for common signal
            mConS = self.mCon[:,iPC]*(1-tSF[ix])
            cConS = self.cCon[:,iPC]*(1-tSF[ix] )

            res[ix] = np.min([nConS-mConS,pConS-mConS])/ np.max(cConS)
        res[res<0] = np.max(res) #set all negative residuals to max so as to bias towards undersubtraction
        optSF = tSF[np.nonzero(np.abs(res)==np.min(np.abs(res)))]

        for iPrec in range(2,10): #define level of precsision required
            tSF = np.ones(19)
            res = np.ones(19)
            for ix in range(-9,10): #serach across 10x required precision of last minimum
                tSF[ix+9] = optSF+ix/10**iPrec
                nConS = self.nCon[:,iPC]-self.cCon[:,iPC]*tSF[ix+9] #- constituent corrected for common signal
                pConS = self.pCon[:,iPC]-self.cCon[:,iPC]*tSF[ix+9] #+ constituent corrected for common signal
                cConS = self.cCon[:,iPC]*(1-tSF[ix+9])
                mConS = self.mCon[:,iPC]*(1-tSF[ix+9])

                res[ix+9] = np.min([nConS-mConS,pConS-mConS])/ np.max(cConS)
            res[res<0] = np.max(res) #set all negative residuals to max so as to bias towards undersubtraction
            optSF = tSF[np.nonzero(np.abs(res)==np.min(np.abs(res)))]
        self.optSF[iPC] = optSF[0]
        self.nConS[:,iPC] = self.nCon[:,iPC]-self.cCon[:,iPC]*optSF #- constituent corrected for common signal
        self.pConS[:,iPC] = self.pCon[:,iPC]-self.cCon[:,iPC]*optSF #+ constituent corrected for common signal
        self.cConS[:,iPC] = self.cCon[:,iPC]*(1-optSF )
        self.mConS[:,iPC] = self.mCon[:,iPC]*(1-optSF )

        ## need to work out how to handle minSpec

        
        #print('extracted positive and negative LEigenvector features')

    def figure_lpniCommonSignalScalingFactors( self , nPC , xcal , xview ):
            ###################       START lpniCommonSignalScalingFactors       #######################
        # FIGURE of the scaling factor calculated for subtracting the common signal from the positive 
        # and negative constituents of a PC
        for iPC in range(1,nPC):
        #generate subtraction figures for positive vs negative score weighted sums.
            self.calc_Constituents( iPC )
            self.figure_lpniLEigenvectorEqn( iPC , xcal )
            self.figure_lpniCommonSignal( iPC , xcal )

        figlpniS, axlpniS = plt.subplots(1,6, figsize=(8,8)) #Extra columns to match spacing in 
        for iPC in range(1,nPC):
            iFig = iPC%6 - 1 # modulo - determine where within block the current PC is
            if iFig==-1:
                iFig = 5 # if no remainder then it is the last in the cycle
            axlpniS[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniS[iFig].plot( self.xcal[[xview[0],xview[-0]]] , [0,0],'--')
            axlpniS[iFig].plot( self.xcal[xview]  , self.nConS[xview,iPC] , 'b', linewidth=1)
            axlpniS[iFig].plot( self.xcal[xview]  , self.pConS[xview,iPC] , 'y', linewidth=1)
            axlpniS[iFig].plot( self.xcal[xview]  , self.cConS[xview,iPC] , 'g', linewidth=0.5)#*smsf[iPC]
            txtpos = [np.mean(axlpniS[iFig].get_xlim()),axlpniS[iFig].get_ylim()[1]*0.9]
            axlpniS[iFig].annotate('PC '+str(iPC+1) , xy=(txtpos) , xytext=(txtpos) , textcoords='data' ,fontsize = 8, horizontalalignment='left')

            if iFig==5:
                for iax in range(np.shape(axlpniS)[0]):
                    axlpniS[iax].axis('off')
                figlpniS.savefig('img\lpniSubLEigenvectorEqn_'+str(iPC-4)+'_'+str(iPC+1)+'.png',dpi=300)
                plt.close()
                figlpniS, axlpniS = plt.subplots(1,6, figsize=(8,8)) #Extra columns to match spacing in 

        plt.figure(figsize=(8,8))
        figsmsf = plt.plot(range(2,np.shape(self.optSF)[0]+1),self.optSF[1:],'.')
        drp = np.add(np.nonzero((self.optSF[2:]-self.optSF[1:-1])<0),2)
        if np.size(drp)!=0:
            plt.plot(drp+1,self.optSF[drp][0],'or')
            plt.plot([2,nPC],[self.optSF[drp][0],self.optSF[drp][0]],'--')
        plt.savefig('img\lpniCommonSignalScalingFactors.png',dpi=300)
        plt.close()

        # copy scalingAdjustment.py into its own cell after running this main cell in Jupyter then you 
        # can manually adjust the scaling factors for each PC to determine what is the most appropriate method

        ###### Plot positive, negative score and common  signals without any  common signal subtraction ######
        figlpniU, axlpniU = plt.subplots(1,6, figsize=(8,8)) #Extra columns to match spacing 
        for iFig in range(6):
            axlpniU[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniU[iFig].plot( self.xcal[[xview[0],xview[-0]]] , [0,0],'--')
            axlpniU[iFig].plot( self.xcal[xview] , self.nCon[xview,iFig+1] , 'b', linewidth=1)
            axlpniU[iFig].plot( self.xcal[xview] , self.pCon[xview,iFig+1] , 'y', linewidth=1)
            axlpniU[iFig].plot( self.xcal[xview] , self.cCon[xview,iFig+1] , 'g', linewidth=0.5)#*smsf[iPC]
            axlpniU[iFig].annotate('PC '+str(iFig+2) , xy=(np.mean(axlpniU[iFig].get_xlim()),axlpniU[iFig].get_ylim()[1]*0.9) , xytext=(np.mean(axlpniU[iFig].get_xlim()),axlpniU[iFig].get_ylim()[1]*0.9) , textcoords='data' ,fontsize = 8, horizontalalignment='left')
        for iax in range(np.shape(axlpniU)[0]):
            axlpniU[iax].axis('off')
        figlpniU.savefig('img\lpniUnSubLEigenvectorEqn_2_4.png',dpi=300)
        plt.close()
        ###################         END lpniCommonSignalScalingFactors           #######################

        
    def figure_lpniLEigenvectorEqn( self , iPC , xcal ):
        # this class function prints out tiff images comparing the score magnitude weighted summed spectra for 
        # positive and negative score spectra. The class must have already calculated the positive, negative 
        # and common consitituents
        if ~np.isnan(self.cCon[0,iPC]):
                  
            figlpni, axlpni = plt.subplots( 1 , 6 , figsize=(8,8))
            axlpni[0] = plt.subplot2grid( (1, 21) , (0, 0) , colspan=1 )
            axlpni[1] = plt.subplot2grid( (1, 21) , (0, 1) , colspan=6 )
            axlpni[2] = plt.subplot2grid( (1, 21) , (0, 7) , colspan=1 )
            axlpni[3] = plt.subplot2grid( (1, 21) , (0, 8) , colspan=6 )
            axlpni[4] = plt.subplot2grid( (1, 21) , (0, 14) , colspan=1 )
            axlpni[5] = plt.subplot2grid( (1, 21) , (0, 15) , colspan=6 )
            posSc = self.LEigenvector[:,iPC]>0 #skip PC1 as not subtraction

            axlpni[0].plot( [-10 ,10] , np.tile(0, (2,1)) , 'k')
            axlpni[0].plot( np.tile( 0 , sum(posSc) ) , self.LEigenvector[posSc,iPC], '.y')
            axlpni[0].plot( np.tile( 0 , sum(posSc==False) ) , self.LEigenvector[posSc==False,iPC] , '.b')

            axlpni[1].plot( xcal , self.X[:,posSc] , 'y')
            axlpni[1].plot( xcal , self.X[:,posSc==False] , '--b' , lw=0.1)
            axlpni[1].annotate( '$s_i$' , xy=(0.1,0.9) , xytext=(xcal[0]-110, 0.9) ,
                           textcoords='data' , fontsize = 20 , horizontalalignment='left')

            axlpni[3].plot( xcal , self.nCon[:,iPC] , 'b')
            axlpni[3].plot( xcal , self.pCon[:,iPC] , 'y')

            pnCon = self.pCon[:,iPC] - self.nCon[:,iPC]
            pnCon = pnCon / np.sum( pnCon ** 2 ) ** 0.5

            axlpni[3].annotate( '$s_i*d_i$' , xy=(0.1,0.9) , xytext=(0.6, 0.9) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
            axlpni[5].plot( self.REigenvector[iPC,:] , 'm' )
            axlpni[5].plot( pnCon , 'c' )
            axlpni[5].plot( self.REigenvector[iPC,:] - pnCon , '--k' )


            ylim = np.max(pnCon)
            axlpni[5].annotate( '$L$' , xy=(0.1,0.9) , xytext=(0, 0.9) , xycoords='axes fraction', 
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='left', 
                               color='m')
            axlpni[5].annotate( '$p-n$' , xy=(0.1,0.9) , xytext=(0, 0.85) ,  xycoords='axes fraction', 
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='left', 
                               color='c')
            axlpni[5].annotate( '$L-(p-n)$' , xy=(0.1,0.9) , xytext=(0, 0.8) ,  xycoords='axes fraction',
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='left', 
                               color='k')
            totdiff = "{:.0f}".format( np.log10( np.mean( np.abs( self.REigenvector[iPC,:] - pnCon ))))
            axlpni[5].annotate( '$\Delta_{L,p-n}=10^{'+ totdiff +'}$' , xy=(0.1,0.9), xytext=(0.1, 0.77) ,
                               xycoords='axes fraction', textcoords='axes fraction' , fontsize = 12 , 
                               horizontalalignment='left', color='k')

            axlpni[2].annotate( '$\Sigma(|s_i^+|*d_i^+)$' , xy=(0,0.5) , xytext=(0.5, 0.55) , 
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
            axlpni[2].annotate( '$\Sigma(|s_i^-|*d_i^-)$', xy=(0,0.5) , xytext=(0.5, 0.42) , 
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
            axlpni[2].annotate( '' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , fontsize = 12 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , 
                                                                          connectionstyle="arc3"))
            axlpni[4].annotate( '$sd_i^+-sd_i^-$' , xy=(0,0.5) , xytext=(0.5, 0.55) , 
                               textcoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
            axlpni[4].annotate( '' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , 
                               fontsize = 12 , horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , 
                                                                          connectionstyle="arc3"))

            for iax in range(np.shape(axlpni)[0]):
                axlpni[iax].axis('off')
            figlpni.savefig( 'img\lpniLEigenvectorEqn ' + str( iPC + 1 ) + '.png' , dpi=300 )
            plt.close()
        else:
            print('Common, Positive and Negative Consituents must be calculated first using calcCons')

    def figure_lpniCommonSignal( self , iPC , xcal ):
        # this class function prints out tiff images comparing the score magnitude weighted summed spectra for 
        # positive and negative score spectra corrected for the common consitituents, compared with the common
        # consituents itself and the scaled global minimum
        
        plt.plot( xcal , self.nConS[:,iPC], 'b')
        plt.plot( xcal , self.pConS[:,iPC],'y')
        plt.plot( xcal , self.cConS[:,iPC],'g')
        plt.plot( xcal , self.minSpec,'c')
        plt.title('PC'+str(iPC)+' Scale Factor:'+str(self.optSF[iPC]))
        plt.legend(('-ve Constituent','+ve Constituent','Common Signal','Global Minimum'))
        plt.savefig('img\lpniDeterminingCommonSignalScalingFactorsPC'+str(iPC+1)+'.png',dpi=300)
        plt.close()