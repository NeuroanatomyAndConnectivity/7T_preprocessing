from nilearn.image import high_variance_confounds
import nibabel as nb
from nilearn.input_data import NiftiMasker
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd


subjects=['']
sessions=['']

base_dir_template='/scr/ilz3/myelinconnect/resting/preprocessed/%s/%s/'
out_dir='/scr/ilz3/myelinconnect/resting/final/'

for subject in subjects:
    for session in sessions:
        
        print 'running '+subject+' '+session
        
        base_dir = base_dir_template%(subject, session)
        moco_file=glob(base_dir+'realignment/corr_'+subject+'*'+session+'_roi.nii.gz')[0]
        brain_mask=glob(base_dir+'mask/'+subject+'*T1_Images_mask_fixed_trans.nii.gz')[0]
        wmcsf_mask=base_dir+'mask/wmcsf_mask_trans.nii.gz'
        motion_file_12=base_dir+'confounds/motion_regressor_der1_ord1.txt'
        artefact_file=glob(base_dir+'confounds/art.corr_'+subject+'*'+session+'_roi_outliers.txt')[0]
        
        
        out_file=out_dir+subject+'_'+session+'_denoised.nii.gz'
        confound_file=base_dir+'confounds/all_confounds.txt'

        # reload niftis to round affines so that nilearn doesn't complain
        wmcsf_nii=nb.Nifti1Image(nb.load(wmcsf_mask).get_data(), np.around(nb.load(wmcsf_mask).get_affine(), 2), nb.load(wmcsf_mask).get_header())
        moco_nii=nb.Nifti1Image(nb.load(moco_file).get_data(),np.around(nb.load(moco_file).get_affine(), 2), nb.load(moco_file).get_header())

        # infer shape of confound array
        confound_len = nb.load(moco_file).get_data().shape[3]
        
        # create outlier regressors
        outlier_regressor = np.empty((confound_len,1))
        try:
            outlier_val = np.genfromtxt(artefact_file)
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((confound_len, 1))
            outlier_vector[index] = 1
            outlier_regressor = np.hstack((outlier_regressor, outlier_vector))
        
        outlier_regressor = outlier_regressor[:,1::]
        
        # load motion regressors
        motion_regressor_12=np.genfromtxt(motion_file_12)
        
        # extract high variance confounds in wm/csf masks from motion corrected data
        wmcsf_regressor=high_variance_confounds(moco_nii, mask_img=wmcsf_nii, detrend=True)
        
        # create Nifti Masker for denoising
        denoiser=NiftiMasker(mask_img=brain_mask, standardize=True, detrend=True, high_pass=0.01, low_pass=0.1, t_r=3.0)
        
        # nilearn wmcsf, moco 12
        confounds=np.hstack((outlier_regressor,wm_regressor, csf_regressor,motion_regressor_12))
        denoised_data=denoiser.fit_transform(moco_file, confounds=confounds)
        denoised_img=denoiser.inverse_transform(denoised_data)
        nb.save(denoised_img, out_file)
        np.savetxt((confound_file), confounds, fmt="%.10f")