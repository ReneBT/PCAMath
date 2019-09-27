import pypandoc
import numpy as np
import nipals
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from matplotlib.patches import ConnectionPatch
from matplotlib import transforms
import pdb

class graphicalPCA:
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many 
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned 
    # along the columns and each row corresponds to different variables
    
    #comments include references to relevant lines in the pseudocode listed in the paper
    
    def __init__( self ):
        ###################            START Data Simulation              #######################
        cwd = os.getcwd() #record current working directory to create relative folder structure

        # Read in simulated fatty acid spectra and associated experimental concentration data
        GCdata = sio.loadmat('D:/Algo/FA profile data.jrb',struct_as_record=False)
        # GC data is from Beattie et al. Lipids 2004 Vol 39 (9):897-906
        FAsim = sio.loadmat('D:/Algo/Papers/Applied Spectroscopy/FA spectra.mat',struct_as_record=False)
        # FAsim are simplified spectra of fatty acid methyl esters built from the properties described in
        # Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        xcal = FAsim['FAXcal'][[0,0]][0,] 
        minSpec = np.tile(np.min(FAsim['simFA'],axis=1),(np.shape(FAsim['simFA'])[1],1))#/np.mean(FAsim['simFA'],axis=1)

        # convert the mass base profile provided into a molar profile
        molarProfile = GCdata['ANNSBUTTERmass']/FAsim['FAproperties'][0,0].MolarMass
        molarProfile = 100.*molarProfile/sum(molarProfile)

        #create a molarprofile with no correlation
        FAmeans = np.mean(molarProfile,1)
        FAsd = np.std(molarProfile,1)
        # generate random numbers in array same size as molarProfile, scale by 1FAsd then add on 
        # mean value
        molarProfileUncorr = np.random.randn(*molarProfile.shape)
        molarProfileUncorr = (molarProfileUncorr.transpose()*FAsd)+FAmeans
        molarProfileUncorr = molarProfileUncorr.transpose()


        #create a molarprofile with only 2 PCs
        nComp = 2
        pca = PCA(nComp)
        pca.fit(molarProfile)
        molarProfile2PC = np.dot(pca.transform(molarProfile)[:,:nComp],pca.components_[:nComp,:])
        molarProfile2PC += pca.mean_

        # Now generate simulated spectra for each sample by multiplying the simualted FA reference 
        # spectra by the FA profiles. Note that the FAsim spectra have a standard intensity in the 
        # carbonyl mode peak (the peak with the highest pixel position)
        spectraFullCov = np.dot(FAsim['simFA'],molarProfile)
        minSpecFullCov = np.dot(np.transpose(minSpec),molarProfile) #will allow scaling of minspec to individual sample
        spectra2Cov = np.dot(FAsim['simFA'],molarProfile2PC)
        spectraNoCov = np.dot(FAsim['simFA'],molarProfileUncorr)
        # Sanity Check by plotting
        plt.plot(FAsim['FAXcal'][[0,0]][0,],spectraFullCov)
        plt.ylabel('Intensity')
        plt.xlabel('Raman Shift cm$^-1$')
        plt.savefig('img\Simulated Data full covariance.png',dpi=300)
        plt.close()


        # Now we calculate PCA, first with the full FA covariance, comparing NIPALS and built in function.

        PCAFullCov = nipals.NIPALS(spectraFullCov,13,10,0.000000000001,'MC', xcal, molarProfile , minSpecFullCov)
        PCAFullCov.calc_PCA()

        PCAFullCovBuiltIn = PCA(n_components=51,) #It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        PCAFullCovBuiltIn.fit(np.transpose(spectraFullCov))
        PCA2CovBuiltIn = PCA(n_components=51,) #It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        PCA2CovBuiltIn.fit(np.transpose(spectra2Cov))

        #test convergence of PCs between NIPALS and SVD based on tolerance
        PCFulldiffTol = np.empty([15,50])
        PC2diffTol = np.empty([15,50])
        for d in range(14):
            tempNIPALS =  nipals.NIPALS(spectraFullCov,51,20,10**-(d+1),'MC') #d starts at 0
            tempNIPALS.calc_PCA()
            for iPC in range(49):
                PCFulldiffTol[d,iPC] =np.log10(np.minimum(np.sum(np.absolute(tempNIPALS.REigenvector[iPC,]-PCAFullCovBuiltIn.components_[iPC,])),np.sum(np.absolute(tempNIPALS.REigenvector[iPC,]+PCAFullCovBuiltIn.components_[iPC,])))) # capture cases of inverted REigenvectors (arbitrary sign switching)
                PC2diffTol[d,iPC] =np.log10(np.minimum(np.sum(np.absolute(tempNIPALS.REigenvector[iPC,]-PCA2CovBuiltIn.components_[iPC,])),np.sum(np.absolute(tempNIPALS.REigenvector[iPC,]+PCAFullCovBuiltIn.components_[iPC,])))) # capture cases of inverted REigenvectors (arbitrary sign switching)

        # compare NIPALs output to builtin SVD based output for 1st PC, switching sign if necessary as this is arbitrary
        plt.plot(FAsim['FAXcal'][[0,0]][0,],PCAFullCov.REigenvector[0,])
        if np.sum(np.absolute(tempNIPALS.REigenvector[0,]-PCAFullCovBuiltIn.components_[0,]))<np.sum(np.absolute(tempNIPALS.REigenvector[iPC,]+PCAFullCovBuiltIn.components_[iPC,])):
            plt.plot(FAsim['FAXcal'][[0,0]][0,],PCAFullCovBuiltIn.components_[0,],':')
        else:
            plt.plot(FAsim['FAXcal'][[0,0]][0,],-PCAFullCovBuiltIn.components_[0,],':')
        plt.ylabel('Weights')
        plt.xlabel('Raman Shift')
        #plt.show()
        plt.close()

        # NIPALS minus builtin SVD based output for 1st PC
        plt.plot(FAsim['FAXcal'][[0,0]][0,],PCAFullCov.REigenvector[0,]+PCAFullCovBuiltIn.components_[0,])
        plt.ylabel('Weights')
        plt.xlabel('Raman Shift')
        #plt.show()
        plt.close()

        #plot the change in difference between NIPALS and SVD against tolerance for PC1
        plt.plot(list(range(-1,-15,-1)),PCFulldiffTol[0:14,3])
        plt.ylabel('$Log_{10}$ of the Absolute Sum of Difference')
        plt.xlabel('$Log_{10}$ NIPALS Tolerance')
        #plt.show()
        plt.close()

        #plot the change in difference between NIPALS and SVD against PC rank for tolerance of 0.001
        plt.plot(list(range(1,49)),PCFulldiffTol[0,0:48],list(range(1,49)),PCFulldiffTol[1,0:48],list(range(1,49)),PCFulldiffTol[2,0:48])
        plt.ylabel('$Log_{10}$ of the Absolute Sum of Difference')
        plt.xlabel('PC Rank')
        plt.close()


        ###################         END  Data Calculations            #######################


        ###################            START DSLTmainEqn              #######################
        #FIGURE for the main PCA equation
        figDSLT, axDSLT = plt.subplots(1,6, figsize=(8,8))
        axDSLT[0] = plt.subplot2grid((5, 20), (0, 0), colspan=8, rowspan=5)
        axDSLT[1] = plt.subplot2grid((5, 20), (0, 8), colspan=1, rowspan=5)
        axDSLT[2] = plt.subplot2grid((5, 20), (0, 9), colspan=2, rowspan=5)
        axDSLT[3] = plt.subplot2grid((5, 20), (0, 11), colspan=1, rowspan=5)
        axDSLT[5] = plt.subplot2grid((5, 20), (0, 12), colspan=8, rowspan=1)
        axDSLT[4] = plt.subplot2grid((5, 20), (1, 12), colspan=8, rowspan=2)
        data4plot = np.empty([spectraFullCov.shape[0],10])
        dataSq4plot = np.empty([spectraFullCov.shape[0],10])
        REigenvectors4plot = np.empty([spectraFullCov.shape[0],5])
        LEigenvectors4plot = np.empty([10,5])
        SSQ = np.sum(PCAFullCov.X**2,1)
        for iDat in range(10):
            data4plot[:,iDat] = PCAFullCov.X[:,iDat]+iDat*16
            dataSq4plot[:,iDat] = PCAFullCov.X[:,iDat]**2+iDat*1000
            LEigenvectors4plot[iDat,:] = PCAFullCov.LEigenvector[iDat,0:5]+iDat*40
        for iDat in range(5):
            REigenvectors4plot[:,iDat] = PCAFullCov.REigenvector[iDat,:]+1-iDat/5
        axDSLT[0].plot(data4plot)
        axDSLT[2].plot(LEigenvectors4plot.transpose(),'.')
        axDSLT[4].plot(REigenvectors4plot,transform=transforms.Affine2D().rotate_deg(90)+plt.gca().transData)
        for iC in range(5):
            axDSLT[4].lines[iC].set_color(str(0+iC/5))

        axDSLT[0].annotate('$D_{-\mu}$', xy=(0,0),xytext=(0.25, 0.9), textcoords='figure fraction',fontsize = 12,horizontalalignment='center')
        axDSLT[0].annotate('$k=1$', xy=(0.08,0.07),xytext=(0.08, 0.2), textcoords='figure fraction',
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05), 
                         fontsize = 12, horizontalalignment='center')
        axDSLT[0].annotate('$k=n$', xy=(0.08,0.06),xytext=(0.08, 0.06), textcoords='figure fraction',
                         fontsize = 12, horizontalalignment='center')
        axDSLT[0].annotate('$j=1$', xy=(0.2,0.04),xytext=(0.1, 0.04), textcoords='figure fraction',  
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize = 12, horizontalalignment='center',va='center')
        axDSLT[0].annotate('$j=p$', xy=(0.2,0.04),xytext=(0.2, 0.04), textcoords='figure fraction',
                         fontsize = 12, horizontalalignment='left',va='center')

        axDSLT[1].annotate('=', xy=(0.5,0.5),xytext=(0.5, 0.5), textcoords='axes fraction',fontsize = 18,horizontalalignment='center')

        axDSLT[2].annotate('$S$', xy=(0.1,0.9),xytext=(0.52, 0.9), textcoords='figure fraction',fontsize = 12,horizontalalignment='left')
        axDSLT[2].annotate('$k=1$', xy=(0.48,0.07),xytext=(0.48, 0.2), textcoords='figure fraction',
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05), 
                         fontsize = 12, horizontalalignment='center')
        axDSLT[2].annotate('$k=n$', xy=(0.48,0.06),xytext=(0.48, 0.06), textcoords='figure fraction',
                         fontsize = 12, horizontalalignment='center')
        axDSLT[2].annotate('$i=1$', xy=(0.6,0.04),xytext=(0.5, 0.04), textcoords='figure fraction',  
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize = 12, horizontalalignment='center',va='center')
        axDSLT[2].annotate('$i=d$', xy=(0.6,0.04),xytext=(0.6, 0.04), textcoords='figure fraction',
                         fontsize = 12, horizontalalignment='left',va='center')


        axDSLT[3].annotate(r'$\cdot$', xy=(0.5,0.5),xytext=(0.5, 0.5),textcoords='axes fraction',fontsize = 32,horizontalalignment='center')

        axDSLT[4].annotate(r'$L{^\top}$', xy=(0.85,0.9),xytext=(0.8,0.9),
                           xycoords='figure fraction',textcoords='figure fraction',fontsize = 12,horizontalalignment='center')
        axDSLT[4].annotate('$j=1$', xy=(0.7,0.07),xytext=(0.7, 0.2), textcoords='figure fraction',
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05), 
                         fontsize = 12, horizontalalignment='center')
        axDSLT[4].annotate('$j=p$', xy=(0.7,0.06),xytext=(0.7, 0.06), textcoords='figure fraction',
                         xycoords='figure fraction',fontsize = 12, horizontalalignment='center')
        axDSLT[4].annotate('$i=1$', xy=(0.85,0.04),xytext=(0.72, 0.04), textcoords='figure fraction',  
                         xycoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize = 12, horizontalalignment='center',va='center')
        axDSLT[4].annotate('$i=d$', xy=(0.85,0.04),xytext=(0.85, 0.04), textcoords='figure fraction',
                         fontsize = 12, horizontalalignment='left',va='center')

        for iax in range(6):
            axDSLT[iax].axis('off')
        figDSLT.savefig('img\DSLTmainEqn.png',dpi=300)
        plt.close()
        ###################             END  DSLTmainEqn              #######################


        ###################            START SLTDscoreEqn             #######################
        #FIGURE for the Scores equation S = L^TD
        figSLTD, axSLTD = plt.subplots(1,6, figsize=(8,8))
        axSLTD[0] = plt.subplot2grid((5, 20), (0, 0), colspan=2, rowspan=5)
        axSLTD[1] = plt.subplot2grid((5, 20), (0, 2), colspan=1, rowspan=5)
        axSLTD[2] = plt.subplot2grid((5, 20), (0, 3), colspan=9, rowspan=1)
        axSLTD[3] = plt.subplot2grid((5, 20), (1, 3), colspan=9, rowspan=2)
        axSLTD[4] = plt.subplot2grid((5, 20), (0, 12), colspan=1, rowspan=5)
        axSLTD[5] = plt.subplot2grid((5, 20), (0, 13), colspan=9, rowspan=5)


        axSLTD[0].plot(LEigenvectors4plot.transpose(),'.')
        axSLTD[3].plot(REigenvectors4plot,transform=transforms.Affine2D().rotate_deg(90)+axSLTD[3].transData)
        axSLTD[5].plot(data4plot)

        axSLTD[0].annotate('$S$', xy=(0.1,0.9),xytext=(0.5, 0.95),
                           textcoords='axes fraction',fontsize = 12,horizontalalignment='center')
        axSLTD[1].annotate('=', xy=(0.5,0.5),xytext=(0.5, 0.5),
                           textcoords='axes fraction',fontsize = 18,horizontalalignment='center')
        axSLTD[2].annotate(r'$L{^\top}$', xy=(0.1,0.9),xytext=(0.5, 0.95),
                           textcoords='axes fraction',fontsize = 12,horizontalalignment='center')
        axSLTD[4].annotate(r'$\cdot$', xy=(0.5,0.5),xytext=(0.5, 0.5),
                           textcoords='axes fraction',fontsize = 38,horizontalalignment='center')
        axSLTD[5].annotate('$D_{-\mu}$', xy=(0.1,0.9),xytext=(0.5, 0.95),
                           textcoords='axes fraction',fontsize = 12,horizontalalignment='center')
        for iax in range(6):
            axSLTD[iax].axis('off')
        figSLTD.savefig('img\SLTDscoreEqn.png',dpi=300)
        plt.close()
        ###################             END  SLTDscoreEqn             #######################


        ###################         START sldiLEigenvectorEqn         #######################
        #FIGURE for the ith LEigenvector equation si = lixD
        figsldi, axsldi = plt.subplots(1,5, figsize=(8,8))
        axsldi[0] = plt.subplot2grid((1, 20), (0, 0), colspan=8)
        axsldi[1] = plt.subplot2grid((1, 20), (0, 8), colspan=1)
        axsldi[2] = plt.subplot2grid((1, 20), (0, 9), colspan=8)
        axsldi[3] = plt.subplot2grid((1, 20), (0, 17), colspan=1)
        axsldi[4] = plt.subplot2grid((1, 20), (0, 18), colspan=2)

        iPC = 1 #the ith PC to plot
        iSamMin = np.argmin(PCAFullCov.LEigenvector[:,iPC-1])
        iSamMax = np.argmax(PCAFullCov.LEigenvector[:,iPC-1])
        iSamZer = np.argmin(np.abs(PCAFullCov.LEigenvector[:,iPC-1]))#Sam = 43 #the ith sample to plot
        sf_iSam = np.mean([sum(PCAFullCov.X[:,iSamMin]**2)**0.5 , sum(PCAFullCov.X[:,iSamMax]**2)**0.5,
                           sum(PCAFullCov.X[:,iSamZer]**2)**0.5]) #use samescaling factor to preserve relative intensity
        offset = np.max(PCAFullCov.REigenvector[iPC-1,:])-np.min(PCAFullCov.REigenvector[iPC-1,:]) #offset for clarity
        axsldi[0].plot(FAsim['FAXcal'][[0,0]][0,], PCAFullCov.REigenvector[iPC-1,:]+offset*1.25, 'k',
                       FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamMax]/sf_iSam +offset/4, 'r',
                       FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamZer]/sf_iSam, 'b',
                       FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamMin]/sf_iSam -offset/4, 'g')
        axsldi[0].legend(('$pc_i$', '$d_{max}$' , '$d_0$', '$d_{min}$' ))
        temp = REigenvectors4plot[:,iPC-1]*PCAFullCov.X[:,iSamZer]
        offsetProd = np.max(temp)-np.min(temp)
        axsldi[2].plot(FAsim['FAXcal'][[0,0]][0,], REigenvectors4plot[:,iPC-1]*PCAFullCov.X[:,iSamMax]+ offsetProd,'r',
                       FAsim['FAXcal'][[0,0]][0,], REigenvectors4plot[:,iPC-1]*PCAFullCov.X[:,iSamZer],'b',
                       FAsim['FAXcal'][[0,0]][0,], REigenvectors4plot[:,iPC-1]*PCAFullCov.X[:,iSamMin] - offsetProd,'g')

        PCilims = np.tile(np.array([np.average(PCAFullCov.LEigenvector[:,iPC-1])-1.96*np.std(PCAFullCov.LEigenvector[:,iPC-1]),
                                    np.average(PCAFullCov.LEigenvector[:,iPC-1]),
                                    np.average(PCAFullCov.LEigenvector[:,iPC-1])+1.96*np.std(PCAFullCov.LEigenvector[:,iPC-1])]),
                          (2,1))
        axsldi[4].plot([0,10],PCilims,'k--',
                       5,PCAFullCov.LEigenvector[iSamMax,iPC-1], 'r.',
                       5,PCAFullCov.LEigenvector[iSamZer,iPC-1],'b.',               
                       5,PCAFullCov.LEigenvector[iSamMin,iPC-1], 'g.', markersize=10)
        ylimLEV = np.abs([PCAFullCov.LEigenvector[:,iPC-1].min(), PCAFullCov.LEigenvector[:,iPC-1].max()]).max()*1.05
        axsldi[4].set_ylim([-ylimLEV, ylimLEV])
        axsldi[1].annotate('', xy=(1,0.5),xytext=(0, 0.5) , xycoords='axes fraction' , fontsize = 12 , horizontalalignment='center' ,
                           arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axsldi[1].annotate(r'$pc_i \times d_i$' , xy=(0.5,0.5) , xytext=(0.5, 0.52) ,
                           xycoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
        axsldi[3].annotate('$\Sigma _{v=1}^{v=p}$' , xy=(0,0.5) , xytext=(0.5, 0.52) ,
                           textcoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
        axsldi[3].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction',
                           fontsize = 18 , horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axsldi[4].annotate('$U95CI$' , xy=(5 , PCilims[0,2]) ,xytext=(10, PCilims[0,2]) ,
                           xycoords = 'data' , textcoords='data' , fontsize = 12 , horizontalalignment='left')
        axsldi[4].annotate('$\overline{S_{i}}$' , xy=(0,0.9) , xytext=(1, 0.49) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='left')
        axsldi[4].annotate('$L95CI$' , xy=(5,PCilims[0,0]) , xytext=(10, PCilims[0,0]) , 
                           xycoords = 'data' , textcoords='data' , fontsize = 12 , horizontalalignment='left')
        for iax in range(5):
            axsldi[iax].axis('off')

        figsldi.savefig('img\sldiLEigenvectorEqn.png',dpi=300)
        plt.close()
        ###################          END  sldiLEigenvectorEqn         #######################


        ###################          Start  sldiResidual         #######################

        figsldiRes, axsldiRes = plt.subplots(1,3, figsize=(8,8))
        axsldiRes[0] = plt.subplot2grid((1, 17), (0, 0), colspan=8)
        axsldiRes[1] = plt.subplot2grid((1, 17), (0, 8), colspan=1)
        axsldiRes[2] = plt.subplot2grid((1, 17), (0, 9), colspan=8)

        iSamResMax = PCAFullCov.X[:,iSamMax]-np.inner(PCAFullCov.LEigenvector[iSamMax,iPC-1],PCAFullCov.REigenvector[iPC-1,:])
        iSamResZer = PCAFullCov.X[:,iSamZer]-np.inner(PCAFullCov.LEigenvector[iSamZer,iPC-1],PCAFullCov.REigenvector[iPC-1,:])
        iSamResMin = PCAFullCov.X[:,iSamMin]-np.inner(PCAFullCov.LEigenvector[iSamMin,iPC-1],PCAFullCov.REigenvector[iPC-1,:])
        offsetRes = np.max(iSamResZer)-np.min(iSamResZer)

        axsldiRes[0].plot(FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamMax]/sf_iSam +offset/4, 'r',
                       FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamZer]/sf_iSam, 'b',
                       FAsim['FAXcal'][[0,0]][0,], PCAFullCov.X[:,iSamMin]/sf_iSam -offset/4, 'g',
                       FAsim['FAXcal'][[0,0]][0,], np.inner(PCAFullCov.LEigenvector[iSamMax,iPC-1],PCAFullCov.REigenvector[iPC-1,:])/sf_iSam +offset/4, 'k--',
                       FAsim['FAXcal'][[0,0]][0,], np.inner(PCAFullCov.LEigenvector[iSamZer,iPC-1],PCAFullCov.REigenvector[iPC-1,:])/sf_iSam, 'k--',
                          FAsim['FAXcal'][[0,0]][0,], np.inner(PCAFullCov.LEigenvector[iSamMin,iPC-1],PCAFullCov.REigenvector[iPC-1,:])/sf_iSam -offset/4, 'k--')
        axsldiRes[0].legend(('$d_{max}$' , '$d_0$' , '$d_{min}$'  , '$pc_i*d_{j}$'))


        axsldiRes[2].plot(FAsim['FAXcal'][[0,0]][0,],iSamResMax+ offsetRes/2,'r',
                       FAsim['FAXcal'][[0,0]][0,],iSamResZer,'b',
                       FAsim['FAXcal'][[0,0]][0,], iSamResMin - offsetRes/2,'g')

        axsldiRes[1].annotate('', xy=(1,0.6),xytext=(0, 0.6) , xycoords='axes fraction' , fontsize = 12 , horizontalalignment='center' ,
                           arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axsldiRes[1].annotate(r'$d_i-(pc_i \times d_i)$' , xy=(0.5,0.5) , xytext=(0.5, 0.62) ,
                           xycoords='axes fraction' , fontsize = 12 , horizontalalignment='center')
        for iax in range(3):
            axsldiRes[iax].axis('off')

        figsldiRes.savefig('img\sldiResLEigenvectorEqn.png',dpi=300)
        plt.close()

        iSamResCorr = np.corrcoef(iSamResMin,iSamResZer)[0,1]**2
        ###################              END sldiResidual            #######################


        # ###################              START LSDscoreEqn            #######################
        # FIGURE for the REigenvectoring equation L = S-1D
        figLSD, axLSD = plt.subplots(1,5, figsize=(8,8))
        axLSD[0] = plt.subplot2grid((1, 20), (0, 0), colspan=8)
        axLSD[1] = plt.subplot2grid((1, 20), (0, 8), colspan=1)
        axLSD[2] = plt.subplot2grid((1, 20), (0, 9), colspan=2)
        axLSD[3] = plt.subplot2grid((1, 20), (0, 11), colspan=1)
        axLSD[4] = plt.subplot2grid((1, 20), (0, 12), colspan=8)

        axLSD[0].plot(REigenvectors4plot)
        axLSD[2].plot(LEigenvectors4plot.transpose(),'.')
        axLSD[4].plot(data4plot)

        axLSD[0].annotate(r'$L{^\top}$' , xy=(0.1,0.95) , xytext=(0.5, 0.95) , textcoords='axes fraction' 
                          , fontsize = 12 , horizontalalignment='center')
        axLSD[1].annotate('=' , xy=(0.5,0.5) , xytext=(0.5, 0.5) , textcoords='axes fraction' , 
                          fontsize = 18 , horizontalalignment='center')
        axLSD[2].annotate('$1/S$' , xy=(0.1,0.95) , xytext=(0.5, 0.95) , textcoords='axes fraction' ,
                          fontsize = 12 , horizontalalignment='center')
        axLSD[3].annotate(r'$\times$' , xy=(0.5,0.5) , xytext=(0.5, 0.5) , textcoords='axes fraction' ,
                          fontsize = 18 , horizontalalignment='center')
        axLSD[4].annotate('$D_{-\mu}$' , xy=(0.1,0.95) , xytext=(0.5, 0.95) , textcoords='axes fraction' ,
                          fontsize = 12 , horizontalalignment='center')
        for iax in range(5):
            axLSD[iax].axis('off')
        figLSD.savefig('img\LSDscoreEqn.png',dpi=300)
        plt.close()
        ###################              END  LSDscoreEqn             #######################


        ###################        START lsdiLEigenvectorEqn          #######################
        # FIGURE for the ith REigenvector equation Li = 1/Si*D
        figlsdi, axlsdi = plt.subplots(1,5, figsize=(8,8))
        axlsdi[0] = plt.subplot2grid((1, 20), (0, 0), colspan=6)
        axlsdi[1] = plt.subplot2grid((1, 20), (0, 6), colspan=1)
        axlsdi[2] = plt.subplot2grid((1, 20), (0, 7), colspan=6)
        axlsdi[3] = plt.subplot2grid((1, 20), (0, 13), colspan=1)
        axlsdi[4] = plt.subplot2grid((1, 20), (0, 14), colspan=6)

        scsf = (PCilims[0,2]-PCilims[0,0])/100 #scale range of LEigenvectors to display beside data
        iPC = 1
        axlsdi[0].plot(np.tile(FAsim['FAXcal'][[0,0]][0,][0]-100,(9,1)) ,
                       PCAFullCov.LEigenvector[0:9,iPC-1],'.')
        axlsdi[0].plot([FAsim['FAXcal'][[0,0]][0,][0]-110 ,
                        FAsim['FAXcal'][[0,0]][0,][0]-90] ,
                       np.tile(PCilims[0,1],(2,1)) , 'k') #mean score for ith PC
        #TODO need different colour per sample
        axlsdi[0].plot(FAsim['FAXcal'][[0,0]][0,] , 1.8*data4plot-250)
        axlsdi[0].annotate('$s_i$', xy=(FAsim['FAXcal'][[0,0]][0,][0]-110,0.9) , xytext=(FAsim['FAXcal'][[0,0]][0,][0]-110, 0.9) , textcoords='data',
                           fontsize = 12,horizontalalignment='left')
        axlsdi[2].plot(PCAFullCov.LEigenvector[0:9,iPC-1]*PCAFullCov.X[:,0:9])
        axlsdi[2].annotate(r'$s_i \times d_i$' , xy=(0.1,0.9) , xytext=(0.6, 0.9) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
        axlsdi[4].plot(PCAFullCov.REigenvector[iPC-1,:])
        axlsdi[4].annotate('$L$' , xy=(0.1,0.9) , xytext=(0.5, 0.9) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')

        axlsdi[1].annotate(r'$\cdot$' , xy=(0,0.5) , xytext=(0.5, 0.52) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
        axlsdi[1].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axlsdi[3].annotate('$\Sigma _{o=1}^{o=n}$' , xy=(0,0.5) , xytext=(0.5, 0.52) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
        axlsdi[3].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , 
                           fontsize = 12 , horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        for iax in range(5):
            axlsdi[iax].axis('off')

        figlsdi.savefig('img\lsdiLEigenvectorEqn.png',dpi=300)
        plt.close()
        ###################              END lsdiLEigenvectorEqn             #######################


        ###################       START lpniCommonSignalScalingFactors       #######################
        # FIGURE of the scaling factor calculated for subtracting the common signal from the positive 
        # and negative constituents of a PC
        nPC = 13
        PCAFullCovNMC = nipals.NIPALS(spectraFullCov,nPC,10,0.000000000001,'NA', xcal, molarProfile , minSpecFullCov)
        PCAFullCovNMC.calc_PCA()
        xview = range(625,685) #zoom in on region to check in detail for changes
        PCAFullCovNMC.figure_lpniCommonSignalScalingFactors( nPC , xview )
        ###################         END lpniCommonSignalScalingFactors           #######################


        ###################                  START DTDscoreEqn                  #######################
        # FIGURE showing how the inner product of the data forms the sum of squares
        figDTD, axDTD = plt.subplots(1,5, figsize=(8,8))
        axDTD[0] = plt.subplot2grid((1, 20), (0, 0), colspan=6)
        axDTD[1] = plt.subplot2grid((1, 20), (0, 6), colspan=1)
        axDTD[2] = plt.subplot2grid((1, 20), (0, 7), colspan=6)
        axDTD[3] = plt.subplot2grid((1, 20), (0, 13), colspan=1)
        axDTD[4] = plt.subplot2grid((1, 20), (0, 14), colspan=6)


        axDTD[0].plot(FAsim['FAXcal'][[0,0]][0,] , data4plot)
        axDTD[2].plot(FAsim['FAXcal'][[0,0]][0,] , dataSq4plot)
        axDTD[2].annotate('$d_i*d_i$' , xy=(0.1,0.9) , xytext=(0.6, 0.9) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')

        axDTD[4].plot(SSQ)
        axDTD[4].annotate('Sum of Squares' , xy=(0.1,0.9) , xytext=(0.5, 0.9) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')

        axDTD[1].annotate('$d_i^2$' , xy=(0,0.5) , xytext=(0.5, 0.55) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
        axDTD[1].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , fontsize = 12 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axDTD[3].annotate('$\Sigma _{o=1}^{o=O}(d_i^2)$' , xy=(0,0.5) , xytext=(0.5, 0.55) , textcoords='axes fraction' ,
                           fontsize = 12 , horizontalalignment='center')
        axDTD[3].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , fontsize = 12 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))

        for iax in range(5):
            axDTD[iax].axis('off')
        figDTD.savefig('img\DTDscoreEqn.png',dpi=300)
        plt.close()
        ###################                  END DTDscoreEqn                  #######################


        ###################                  START D2DwscoreEqn               #######################
        # FIGURE for the illustration of the NIPALs algorithm, with the aim of iteratively calculating 
        # each PCA to minimise the explantion of the sum of squares
        figD2Dw, axD2Dw = plt.subplots(3,5, figsize=(8,8))
        axD2Dw[0,0] = plt.subplot2grid((13, 20), (0, 0), colspan=6, rowspan=6)
        axD2Dw[0,1] = plt.subplot2grid((13, 20), (0, 6), colspan=1, rowspan=6)
        axD2Dw[0,2] = plt.subplot2grid((13, 20), (0, 7), colspan=6, rowspan=6)
        axD2Dw[0,3] = plt.subplot2grid((13, 20), (0, 13), colspan=1, rowspan=6)
        axD2Dw[0,4] = plt.subplot2grid((13, 20), (0, 14), colspan=6, rowspan=6)
        axD2Dw[1,0] = plt.subplot2grid((13, 20), (7, 0), colspan=7, rowspan=1)
        axD2Dw[1,1] = plt.subplot2grid((13, 20), (7, 7), colspan=7, rowspan=1)
        axD2Dw[1,2] = plt.subplot2grid((13, 20), (7, 14), colspan=6, rowspan=1)
        axD2Dw[2,0] = plt.subplot2grid((13, 20), (8, 0), colspan=6, rowspan=6)
        axD2Dw[2,1] = plt.subplot2grid((13, 20), (8, 6), colspan=1, rowspan=6)
        axD2Dw[2,2] = plt.subplot2grid((13, 20), (8, 7), colspan=6, rowspan=6)
        axD2Dw[2,3] = plt.subplot2grid((13, 20), (8, 13), colspan=1, rowspan=6)
        axD2Dw[2,4] = plt.subplot2grid((13, 20), (8, 14), colspan=6, rowspan=6)

        axD2Dw[0,0].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.r[0])
        ylims0_0 = np.max(np.abs(axD2Dw[0,0].get_ylim()))
        axD2Dw[0,0].set_ylim(-ylims0_0,ylims0_0) #tie the y limits so scales directly comparable
        axD2Dw[0,0].annotate('a) $D_{-\mu}=R_0$', xy=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,0].get_ylim()[1])  , xytext=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,0].get_ylim()[1]) , textcoords='data',
                           fontsize = 8,horizontalalignment='left')
        axD2Dw[0,1].annotate('$\widehat{\Sigma(R_0^2)}$' , xy=(0,0.5) , xytext=(0.5, 0.55) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center')
        axD2Dw[0,1].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , fontsize = 8 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))

        axD2Dw[0,1].annotate('$j=0$' , xy=(0,0.5) , xytext=(0.5, 0.45) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center')
        axD2Dw[0,2].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.pc[1][:,0],'m')
        axD2Dw[0,2].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.pc[0][:,0])
        axD2Dw[0,2].annotate('b) $\widehat{SS_R}$', xy=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,2].get_ylim()[1])  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,2].get_ylim()[1]) , textcoords='data',
                             fontsize = 8,horizontalalignment='left')
        axD2Dw[0,3].annotate('$R_{i-1}/\widehat{SS}$' , xy=(0,0.5) , xytext=(0.5, 0.55) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center')
        axD2Dw[0,3].annotate('' , xy=(1,0.5) , xytext=(0, 0.5) , textcoords='axes fraction' , fontsize = 8 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axD2Dw[0,3].annotate('$i=i+1$' , xy=(0,0.5) , xytext=(0.5, 0.45) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center')
        axD2Dw[0,3].annotate('$j=j+1$' , xy=(0,0.5) , xytext=(0.5, 0.40) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center')
        axD2Dw[0,4].plot(PCAFullCov.w[1][0,:],'.m')
        axD2Dw[0,4].plot(PCAFullCov.w[0][0,:]/10,'.')
        axD2Dw[0,4].plot(PCAFullCov.w[0][1,:],'.c')
        axD2Dw[0,4].annotate('c) $S_i^j$', xy=(8, axD2Dw[0,4].get_ylim()[1])  , 
                             xytext=(8, axD2Dw[0,4].get_ylim()[1]) , textcoords='data',
                             fontsize = 8,horizontalalignment='left')
        axD2Dw[1,2].annotate('' , xy=(0.5,0) , xytext=(0.5, 1) , textcoords='axes fraction' , fontsize = 8 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axD2Dw[1,2].annotate('$D_{-\mu}/S_i^j$', xy=(0.55,0.5)  , xytext=(0.55, 0.55) , textcoords='axes fraction',
                             fontsize = 8,horizontalalignment='centre', rotation = 90)
        axD2Dw[2,4].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.pc[1][:,1],'m')
        axD2Dw[2,4].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.pc[0][:,1])
        axD2Dw[2,4].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.pc[0][:,2],'c')
        ylims2_4 = np.max(np.abs(axD2Dw[2,4].get_ylim()))
        axD2Dw[2,4].set_ylim(-ylims2_4,ylims2_4) #tie the y limits so scales directly comparable
        axD2Dw[2,4].annotate('d) $L_i^{Tj}$', xy=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[2,4].get_ylim()[0])  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[2,4].get_ylim()[0]) , textcoords='data',
                             fontsize = 8,horizontalalignment='left')
        axD2Dw[2,3].annotate('' , xy=(0,0.5) , xytext=(1, 0.5) , textcoords='axes fraction' , fontsize = 8 ,
                           horizontalalignment='center' , arrowprops=dict(arrowstyle="->" , connectionstyle="arc3"))
        axD2Dw[2,3].annotate('$|L_i^{jT}-L_i^{(j-1)T}|$', xy=(0.5, 0.55)  , 
                             xytext=(0.5,0.55) , textcoords='axes fraction',
                             fontsize = 8,horizontalalignment='center')
        axD2Dw[2,2].plot(FAsim['FAXcal'][[0,0]][0,] , np.abs(PCAFullCov.pc[1][:,1]-PCAFullCov.pc[1][:,0]),'m')
        axD2Dw[2,2].plot(FAsim['FAXcal'][[0,0]][0,] , np.abs(PCAFullCov.pc[0][:,1]-PCAFullCov.pc[0][:,0]))
        ylims2_2 = np.max(np.abs(axD2Dw[2,2].get_ylim()))*1.1
        axD2Dw[2,2].plot(FAsim['FAXcal'][[0,0]][0,] , np.abs(PCAFullCov.pc[0][:,2]-PCAFullCov.pc[0][:,1]),'c')
        axD2Dw[2,2].set_ylim([0-ylims2_2*0.1,ylims2_2]) 
        axD2Dw[2,2].annotate('e) Iteration Change in $L^T$', xy=(FAsim['FAXcal'][[0,0]][0,][10], 0-ylims2_2*0.1)  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][10], 0-ylims2_2*0.1) , textcoords='data',
                             fontsize = 8,horizontalalignment='left')
        axD2Dw[2,2].annotate('$\Sigma|L_i^{jT}-L_i^{(j-1)T}|<Tol$ OR $j=max\_j$', xy=(FAsim['FAXcal'][[0,0]][0,][200], ylims2_2*0.9)  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][10], ylims2_2*0.87) , textcoords='data',
                             fontsize = 8,horizontalalignment='centre')
        axD2Dw[2,2].annotate('$True$', xy=(FAsim['FAXcal'][[0,0]][0,][0], ylims2_2*0.85)  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][50], ylims2_2*0.79) , textcoords='data',
                             fontsize = 8,horizontalalignment='left', color = 'g')
        axD2Dw[2,2].annotate('$False$', xy=(FAsim['FAXcal'][[0,0]][0,][400], ylims2_2*0.99)  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][450], ylims2_2*0.95) , textcoords='data',
                             fontsize = 8,horizontalalignment='right', color = 'r')
        con = ConnectionPatch(xyA=(FAsim['FAXcal'][[0,0]][0,][100],ylims2_2*0.79), xyB=(FAsim['FAXcal'][[0,0]][0,][-1],ylims0_0*0.1), coordsA="data", coordsB="data",
                              axesA=axD2Dw[2,2], axesB=axD2Dw[2,0],arrowstyle="->")
        axD2Dw[2,2].add_artist(con)
        con2 = ConnectionPatch(xyA=(FAsim['FAXcal'][[0,0]][0,][450],ylims2_2), xyB=(axD2Dw[0,4].get_xlim()[0],axD2Dw[0,4].get_ylim()[0]), coordsA="data", coordsB="data",
                              axesA=axD2Dw[2,2], axesB=axD2Dw[0,4],arrowstyle="->")
        axD2Dw[2,2].add_artist(con2)
        axD2Dw[1,1].annotate('$R_{i-1}^T*L_i^j}$' , xy=(0,0.5) , xytext=(0.7, 1) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='right', rotation = 42)
        axD2Dw[1,1].annotate('$j=j+1$' , xy=(0,0.5) , xytext=(0.75, 0.95) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center', rotation = 42)
        axD2Dw[2,1].annotate('$R_{i-1}-S_{i}*L_{i}^T$' , xy=(0.1,0.9) , xytext=(0.75, 0.82) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center', rotation = 40)
        axD2Dw[2,0].plot(FAsim['FAXcal'][[0,0]][0,] , PCAFullCov.r[1])
        axD2Dw[2,0].set_ylim(-ylims0_0,ylims0_0) #tie the y limits so scales directly comparable
        axD2Dw[2,0].annotate('f) $R_i$', xy=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,0].get_ylim()[0])  , 
                             xytext=(FAsim['FAXcal'][[0,0]][0,][10], axD2Dw[0,0].get_ylim()[0]) , textcoords='data',
                             fontsize = 8,horizontalalignment='left')
        con3 = ConnectionPatch(xyA=(FAsim['FAXcal'][[0,0]][0,][450],np.max(PCAFullCov.r[1]*2)), xyB=(FAsim['FAXcal'][[0,0]][0,][0],0), coordsA="data", coordsB="data",
                              axesA=axD2Dw[2,0], axesB=axD2Dw[0,2],arrowstyle="->")
        axD2Dw[2,0].add_artist(con3)
        axD2Dw[1,0].annotate('$\widehat{\Sigma(R_i^2)}$' , xy=(0,0.5) , xytext=(0.75, 0.9) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center', rotation = 55)
        axD2Dw[1,0].annotate('$j=0$' , xy=(0,0.5) , xytext=(0.85, 0.5) , textcoords='axes fraction' ,
                           fontsize = 8 , horizontalalignment='center', rotation = 55)
        for iax in range(5):
            axD2Dw[0,iax].axis('off')
            axD2Dw[2,iax].axis('off')
            if iax<3:
                axD2Dw[1,iax].axis('off')    
        figD2Dw.subplots_adjust(wspace=0,hspace=0)
        figD2Dw.savefig('img\D2DwscoreEqn.png',dpi=300)
        plt.close()
        ###################                  END D2DwscoreEqn                  #######################
        return