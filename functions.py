def selectindex(files, idx):
    import numpy as np
    from nipype.utils.filemanip import filename_to_list, list_to_filename
    return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

def fix_hdr(data_file, header_file):
    
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    
    data=nb.load(data_file).get_data()
    hdr=nb.load(header_file).get_header()
    affine=nb.load(header_file).get_affine()
    
    new_file=nb.Nifti1Image(data, affine, hdr)
    _, base, _ = split_filename(data_file)
    nb.save(new_file, base + "_fixed.nii.gz")
    return os.path.abspath(base + "_fixed.nii.gz")
    

def median(in_files):
    """Computes an average of the median of each realigned timeseries
    Parameters
    ----------
    in_files: one or more realigned Nifti 4D time series
    Returns
    -------
    out_file: a 3D Nifti file
    """
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import filename_to_list
    from nipype.utils.filemanip import split_filename
    
    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.get_affine(), img.get_header())
    #filename = os.path.join(os.getcwd(), 'median.nii.gz')
    #median_img.to_filename(filename)
    _, base, _ = split_filename(filename_to_list(in_files)[0])
    nb.save(median_img, base + "_median.nii.gz")
    return os.path.abspath(base + "_median.nii.gz")
    return filename



def get_info(dicom_file):
    """Given a Siemens dicom file return metadata
    Returns
    -------
    RepetitionTime
    Slice Acquisition Times
    Spacing between slices
    """
    from dcmstack.extract import default_extractor
    import numpy as np
    from dicom import read_file
    from nipype.utils.filemanip import filename_to_list
    
    meta = default_extractor(read_file(filename_to_list(dicom_file)[0],
                                       stop_before_pixels=True,
                                       force=True))
    
    TR=meta['RepetitionTime']/1000.
    slice_times_pre=meta['CsaImage.MosaicRefAcqTimes']
    slice_times = (np.array(slice_times_pre)/1000.).tolist()
    slice_thickness = meta['SpacingBetweenSlices']
    
    return TR, slice_times, slice_thickness


def strip_rois_func(in_file, t_min):
    import numpy as np
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    nii = nb.load(in_file)
    new_nii = nb.Nifti1Image(nii.get_data()[:,:,:,t_min:], nii.get_affine(), nii.get_header())
    new_nii.set_data_dtype(np.float32)
    _, base, _ = split_filename(in_file)
    nb.save(new_nii, base + "_roi.nii.gz")
    return os.path.abspath(base + "_roi.nii.gz")


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os
    
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor_der%d_ord%d.txt" % (derivatives, order))
        np.savetxt(filename, out_params2, fmt="%.10f")
        out_files.append(filename)
    return out_files
