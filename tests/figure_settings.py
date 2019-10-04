# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file to test the figure settings functionality
"""
                    ############### start of setup ######################
import numpy as np
import scipy.io as sio
import os

import src.local_nipals as npls

# This expects to be called inside the jupyter project folder structure.
cwd = (
     os.getcwd()
 )
GC_data = sio.loadmat(
    cwd[:-5] + "data\\FA profile data.jrb", struct_as_record=False
)
simplified_fatty_acid_spectra = sio.loadmat(cwd[:-5] + "data\\FA spectra.mat", struct_as_record=False)
wavelength_axis = simplified_fatty_acid_spectra["FAXcal"][[0, 0]][0,]
min_spectral_values = np.tile(
    np.min(simplified_fatty_acid_spectra["simFA"], axis=1), (np.shape(simplified_fatty_acid_spectra["simFA"])[1], 1)
)  
molar_profile = GC_data["ANNSBUTTERmass"] / simplified_fatty_acid_spectra["FAproperties"][0, 0].MolarMass
molar_profile = 100.0 * molar_profile / sum(molar_profile)
spectra_full_covariance = np.dot(simplified_fatty_acid_spectra["simFA"], molar_profile)
min_spectra_full_covariance = np.dot(
    np.transpose(min_spectral_values), molar_profile
)  
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
                    ############### END of setup ######################


                    ############### Start of testing ######################

#pca_full_covariance.figure_Settings(size=[9,9], res = [400] , frmt = 'dfs', k = [1,4,6] , i= [3,4,6], Ylab = 'Testing', Xlab = 'TESTING A REALLY LONG STRING THAT IS HOPEFULLY LONG ENOUGH TO NEED TRUNCATION', Show_Labels = False , Show_Values = True, TxtSz = 15)
pca_full_covariance.figure_Settings(size=[22,3], res = [400] , frmt = 'jpg', k = [1,4,6] , i= [3,4,6], Ylab = 'TESTING A REALLY LONG STRING THAT IS HOPEFULLY LONG', Xlab = 'TESTING A SLIGHTLY LONG STRING', Show_Labels = False , Show_Values = 3, TxtSz = 21, Project = 'Meh blah blah')

print('Size: '+str(pca_full_covariance.fig_Size))
print('Resolution: : '+str(pca_full_covariance.fig_Resolution))
print('Format: '+str(pca_full_covariance.fig_Format))
print('k: '+str(pca_full_covariance.fig_k))
print('i: '+str(pca_full_covariance.fig_i))
print('Y_Label: '+str(pca_full_covariance.fig_Y_Label))
print('X_Label: '+str(pca_full_covariance.fig_X_Label))
print('Show_Labels: '+str(pca_full_covariance.fig_Show_Labels))
print('Show_Values: '+str(pca_full_covariance.fig_Show_Values))
print('Text_Size: '+str(pca_full_covariance.fig_Text_Size))
print('Project Code: '+str(pca_full_covariance.fig_Project))


