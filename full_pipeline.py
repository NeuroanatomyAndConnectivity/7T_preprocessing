from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.nipy as nipy
import nipype.interfaces.afni as afni
import nipype.algorithms.rapidart as ra
from nipype.algorithms.misc import TSNR
import nipype.interfaces.ants as ants
from functions import strip_rois_func, get_info, median, motion_regressors, selectindex, fix_hdr, nilearn_denoise
from linear_coreg import create_coreg_pipeline
from nonlinear_coreg import create_nonlinear_pipeline


# FREESURFER 
# FSL 
# AFNI 
# ANTSENV
# C3D 
# DCMSTACK
# NILEARN
# source /scr/animals1/nipype_env/bin/activate

'''
------
Inputs
------
'''

# read in subjects and file names
subjects=['sub001'] #, 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 
          # 'sub007', 'sub008', 'sub009', 'sub010', 'sub011', 'sub012', 
          # 'sub013', 'sub014', 'sub015', 'sub016', 'sub017', 'sub018', 
          # 'sub019', 'sub020', 'sub021', 'sub022']
# sessions to loop over
sessions=['session_1' ,'session_2']
# scans to loop over
scans=['rest_full_brain_1', 'rest_full_brain_2']

# directories
working_dir = '/scr/animals1/preproc7t/working_dir_test/'
data_dir= '/scr/animals1/preproc7t/data7t/'
out_dir = '/scr/animals1/preproc7t/resting/preprocessed/'
freesurfer_dir = '/scr/animals1/preproc7t/freesurfer/' 

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# number of processes to run in parallel with multiproc
n_proc = 1

# number of volumes to remove from the resting state timeseries
n_vol_remove = 5


'''
------------------
Construct workflow
------------------
'''

preproc = Workflow(name='func_preproc')
preproc.base_dir = working_dir
preproc.config['execution']['crashdump_dir'] = preproc.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# iterate over scans
scan_infosource = Node(util.IdentityInterface(fields=['scan']), 
                  name='scan_infosource')
scan_infosource.iterables=[('scan', scans)]

# select files
templates={'rest' : 'niftis/{subject}/{session}/{scan}.nii.gz',
           'dicom':'dicoms_example/MR.2.25.130666515827674933471189335089197862909.dcm',
           'uni_highres' : 'niftis/{subject}/{session}/MP2RAGE_UNI.nii.gz', # changed to lowres
           't1_highres' : 'niftis/{subject}/{session}/MP2RAGE_T1.nii.gz', # changed to lowres
           'brain_mask' : 'brainmasks/{subject}/ses-1/anat/*.nii.gz',
           }

selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

preproc.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')]), 
                 (scan_infosource, selectfiles, [('scan', 'scan')]),  
                 ])



'''
------------------------
Structural Preprocessing
------------------------
'''

# fix header of supplied brain mask
fixhdr = Node(util.Function(input_names=['data_file', 'header_file'],
                            output_names=['out_file'],
                            function=fix_hdr),
                  name='fixhdr')
preproc.connect([(selectfiles, fixhdr, [('brain_mask', 'data_file'),
                                        ('uni_highres', 'header_file')]),
                 ])

# mask uni image with fixed brain mask
mask_uni = Node(fsl.ApplyMask(),name='mask_uni')
preproc.connect([(fixhdr, mask_uni, [('out_file', 'mask_file')]),
                 (selectfiles, mask_uni,[('uni_highres', 'in_file')])
                ])

# run reconall
#recon_all = Node(fs.ReconAll(args='-nuiterations 7 -no-isrunning'),
#                 name="recon_all")
#recon_all.plugin_args={'submit_specs': 'request_memory = 9000'}
#recon_all.inputs.subjects_dir=freesurfer_dir

# function to replace / in subject id string with a _
def sub_id(sub_id):
    return sub_id.replace('/','_')

#preproc.connect([(mask_uni, recon_all, [('out_file', 'T1_files')]),
#                  (subject_infosource, recon_all, [(('subject', sub_id), 'subject_id')])
#                  ])
 
# Grab brain and T1 file and segmentation from Freesurfer 
# to store denoised T1, create more precise brainmask and wmcsfmask
fs_import = Node(interface=nio.FreeSurferSource(),
                 name = 'fs_import')

preproc.connect([#(recon_all, fs_import, [('subject_id', 'subject_id'),
                 #                      ('subjects_dir', 'subjects_dir')])
                 (subject_infosource, fs_import, [(('subject', sub_id), 'subject_id')])
                 ])
fs_import.inputs.subjects_dir=freesurfer_dir

# convert Freesurfer T1 file to nifti
head_convert=Node(fs.MRIConvert(out_type='niigz',
                                 out_file='uni_lowres.nii.gz'),
                   name='head_convert')

preproc.connect([(fs_import, head_convert, [('T1', 'in_file')])])


def get_aparc_aseg(files):
    for name in files:
        if 'aparc+aseg' in name:
            return name

# create brainmask from aparc+aseg with single dilation
brainmask = Node(fs.Binarize(min=0.5,
                             dilate=1,
                             out_type='nii.gz'),
               name='brainmask')

preproc.connect([(fs_import, brainmask, [(('aparc_aseg', get_aparc_aseg), 'in_file')])])

# fill holes in mask
fillholes = Node(fsl.maths.MathsCommand(args='-fillh -s 3 -thr 0.1 -bin',
                                        out_file='uni_lowres_brain_mask.nii.gz'),
                 name='fillholes')
preproc.connect([(brainmask, fillholes, [('binary_file', 'in_file')])])

# create wmcsf mask
wm_csf_mask = Node(fs.Binarize(wm_ven_csf = True,
                          erode = 2,
                          out_type = 'nii.gz',
                          binary_file='wmcsf_mask.nii.gz'), 
               name='wm_csf_mask')

preproc.connect([(fs_import, wm_csf_mask, [(('aparc_aseg', get_aparc_aseg), 'in_file')])
                 ])



'''
------------------------
Functional Preprocessing
------------------------
'''

# remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = n_vol_remove

preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# get slice time information from example dicom
getinfo = Node(util.Function(input_names=['dicom_file'],
                             output_names=['TR', 'slice_times', 'slice_thickness'],
                             function=get_info),
               name='getinfo')
preproc.connect([(selectfiles, getinfo, [('dicom', 'dicom_file')])])
                 
                 
# simultaneous slice time and motion correction
slicemoco = Node(nipy.SpaceTimeRealigner(),                 
                 name="spacetime_realign")
slicemoco.inputs.slice_info = 2

preproc.connect([(getinfo, slicemoco, [('slice_times', 'slice_times'),
                                       ('TR', 'tr')]),
                 (remove_vol, slicemoco, [('out_file', 'in_file')])])

# compute tsnr and detrend
tsnr = Node(TSNR(regress_poly=2),
               name='tsnr')
preproc.connect([(slicemoco, tsnr, [('out_file', 'in_file')])])
 
# compute median of realigned timeseries for coregistration to anatomy
median = Node(util.Function(input_names=['in_files'],
                       output_names=['median_file'],
                       function=median),
              name='median')
 
preproc.connect([(tsnr, median, [('detrended_file', 'in_files')])])
 
# make FOV mask for later nonlinear coregistration
fov = Node(fsl.maths.MathsCommand(args='-bin',
                                  out_file='fov_mask.nii.gz'),
           name='fov_mask')
preproc.connect([(median, fov, [('median_file', 'in_file')])])


# biasfield correction of median epi for better registration
biasfield = Node(ants.segmentation.N4BiasFieldCorrection(save_bias=True),
                 name='biasfield')
preproc.connect([(median, biasfield, [('median_file', 'input_image')])])

# perform linear coregistration in ONE step: median2lowres
coreg=create_coreg_pipeline()
coreg.inputs.inputnode.fs_subjects_dir = freesurfer_dir
 
preproc.connect([(selectfiles, coreg, [('uni_highres', 'inputnode.uni_highres')]),
                 (head_convert, coreg, [('out_file', 'inputnode.uni_lowres')]),
                 (biasfield, coreg, [('output_image', 'inputnode.epi_median')]),
                 (subject_infosource, coreg, [('subject', 'inputnode.fs_subject_id')])
                ])

# perform nonlinear coregistration 
nonreg=create_nonlinear_pipeline()
   
preproc.connect([(selectfiles, nonreg, [('t1_highres', 'inputnode.t1_highres')]),
                 (fillholes, nonreg, [('out_file', 'inputnode.brain_mask')]),
                 (wm_csf_mask, nonreg, [('binary_file', 'inputnode.wmcsf_mask')]),
                 (fov, nonreg, [('out_file', 'inputnode.fov_mask')]),
                 (coreg, nonreg, [('outputnode.epi2highres_lin', 'inputnode.epi2highres_lin'),
                                  ('outputnode.epi2highres_lin_itk', 'inputnode.epi2highres_lin_itk'),
                                  ('outputnode.highres2lowres_itk', 'inputnode.highres2lowres_itk')])
                  ])

# merge struct2func transforms into list
translist_inv = Node(util.Merge(2),name='translist_inv')
preproc.connect([(coreg, translist_inv, [('outputnode.epi2highres_lin_itk', 'in1')]),
                 (nonreg, translist_inv, [('outputnode.epi2highres_invwarp', 'in2')])])
   
# merge images into list
structlist = Node(util.Merge(2),name='structlist')
preproc.connect([(nonreg, structlist, [('outputnode.brainmask_highres', 'in1'),
                                      ('outputnode.wmcsfmask_highres', 'in2')]),             
                 ])
   
# project brain mask and wm/csf masks in functional space
struct2func = MapNode(ants.ApplyTransforms(dimension=3,
                                         invert_transform_flags=[True, False],
                                         interpolation = 'NearestNeighbor'),
                    iterfield=['input_image'],
                    name='struct2func')

   
preproc.connect([(structlist, struct2func, [('out', 'input_image')]),
                 (translist_inv, struct2func, [('out', 'transforms')]),
                 (median, struct2func, [('median_file', 'reference_image')]),
                 ])


# perform artefact detection
artefact=Node(ra.ArtifactDetect(save_plot=True,
                                use_norm=True,
                                parameter_source='NiPy',
                                mask_type='file',
                                norm_threshold=1,
                                zintensity_threshold=3,
                                use_differences=[True,False]
                                ),
             name='artefact')
   
preproc.connect([(slicemoco, artefact, [('out_file', 'realigned_files'),
                                        ('par_file', 'realignment_parameters')]),
                 (struct2func, artefact, [(('output_image', selectindex, [0]), 'mask_file')]),
                 ])
  
# calculate motion regressors
motreg = Node(util.Function(input_names=['motion_params', 'order','derivatives'],
                            output_names=['out_files'],
                            function=motion_regressors),
                 name='motion_regressors')
motreg.inputs.order=1
motreg.inputs.derivatives=1
preproc.connect([(slicemoco, motreg, [('par_file','motion_params')])])
  
# use Nilearn to calculate physiological nuissance regressors and clean 
# time series using combined regressors
denoise = Node(util.Function(input_names=['in_file', 
                                          'brain_mask', 'wm_csf_mask',
                                          'motreg_file', 
                                          'outlier_file', 
                                          'bandpass', 
                                          'tr'],
                             output_names=['denoised_file',
                                           'confounds_file'],
                             function=nilearn_denoise),
               name='denoise')

denoise.inputs.tr = 3.0
denoise.inputs.bandpass = [0.1, 0.01]

preproc.connect([(slicemoco, denoise, [('out_file', 'in_file')]),
                 (struct2func, denoise, [(('output_image', selectindex, [0]), 'brain_mask'),
                                         (('output_image', selectindex, [1]), 'wm_csf_mask')]),
                 (motreg, denoise, [(('out_files',selectindex,[0]), 'motreg_file')]),
                 (artefact, denoise, [('outlier_files', 'outlier_file')])
                 ])

  
'''
-------
Outputs
-------
'''  


def make_basedir(out_dir, subject, session, scan):
    return out_dir+sub+'/'+sess+'/'+scan+'/'

makebase = Node(util.Function(input_names=['out_dir','subject',
                                           'session', 'scan'],
                             output_names=['base_dir'],
                             function=make_basedir),
               name='makebase')
    
preproc.connect([(subject_infosource, makebase, [('subject', 'subject')]),
                 (session_infosource, makebase, [('session', 'session')]),
                 (scan_infosource, makebase, [('scan', 'scan')])
                 ])
    
    
sink = Node(nio.DataSink(parameterization=True),
             name='sink')

preproc.connect([(makebase, sink, [('base_dir', 'base_directory')]),
                 (head_convert, sink, [('out_file', 'registration.@anat_head')]),
                 (fillholes, sink, [('out_file', 'registration.@brain_mask')]),
                 (remove_vol, sink, [('out_file', 'realignment.@raw_file')]),
                 (slicemoco, sink, [('out_file', 'realignment.@realigned_file'),
                                    ('par_file', 'confounds.@orig_motion')]),
                 (tsnr, sink, [('tsnr_file', 'realignment.@tsnr')]),
                 (median, sink, [('median_file', 'realignment.@median')]),
                 (biasfield, sink, [('output_image', 'realignment.@biasfield')]),
                 (coreg, sink, [('outputnode.epi2lowres', 'registration.@epi2lowres'),
                                ('outputnode.epi2lowres_mat','registration.@epi2lowres_mat'),
                                ('outputnode.epi2lowres_dat','registration.@epi2lowres_dat'),
                                ('outputnode.highres2lowres', 'registration.@highres2lowres'),
                                ('outputnode.highres2lowres_dat', 'registration.@highres2lowres_dat'),
                                ('outputnode.highres2lowres_mat', 'registration.@highres2lowres_mat'),
                                ('outputnode.epi2highres_lin', 'registration.@epi2highres_lin'),
                                ('outputnode.epi2highres_lin_itk', 'registration.@epi2highres_lin_itk'),
                                ('outputnode.epi2highres_lin_mat', 'registration.@epi2highres_lin_mat')]),
                (nonreg, sink, [('outputnode.epi2highres_warp', 'registration.@epi2highres_warp'),
                                ('outputnode.epi2highres_invwarp', 'registration.@epi2highres_invwarp'),
                                ('outputnode.epi2highres_nonlin', 'registration.@epi2highres_nonlin')]),
                (struct2func, sink, [(('output_image', selectindex, [0,1]), 'mask.@masks')]),
                (artefact, sink, [('norm_files', 'confounds.@norm_motion'),
                                  ('outlier_files', 'confounds.@outlier_files'),
                                  ('intensity_files', 'confounds.@intensity_files'),
                                  ('statistic_files', 'confounds.@outlier_stats'),
                                  ('plot_files', 'confounds.@outlier_plots')]),
                 (motreg, sink, [('out_files', 'confounds.@motreg')]),
                 (denoise, sink, [('denoised_file', 'final.@final'),
                                  ('confounds_file', 'confounds.@all')])
                 ])


'''
------------
Run workflow
------------
'''

#preproc.run()
#plugin='MultiProc', plugin_args={'n_procs' : n_proc})
preproc.run(plugin='CondorDAGMan', plugin_args = {'override_specs': 'request_memory = 20000'})
