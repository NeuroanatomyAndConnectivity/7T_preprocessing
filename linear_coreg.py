from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.c3 as c3
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.ants as ants
import nipype.interfaces.io as nio
import os

def create_coreg_pipeline(name='coreg'):
    
    # fsl output type
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # initiate workflow
    coreg = Workflow(name='coreg')
    
    #inputnode 
    inputnode=Node(util.IdentityInterface(fields=['epi_median',
                                                  'fs_subjects_dir',
                                                  'fs_subject_id',
                                                  'uni_lowres',
                                                  ]),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['uni_lowres',
                                                   'epi2lowres_lin',
                                                   'epi2lowres_lin_mat',
                                                   'epi2lowres_lin_dat',
                                                   'epi2lowres_lin_itk',
                                                   ]),
                    name='outputnode')
    
    
    
    # convert mgz head file for reference
    fs_import = Node(interface=nio.FreeSurferSource(),
                     name = 'fs_import')
    
    brain_convert=Node(fs.MRIConvert(out_type='niigz', 
                                     out_file='uni_lowres.nii.gz'),
                       name='brain_convert')
    
    coreg.connect([(inputnode, fs_import, [('fs_subjects_dir','subjects_dir'),
                                            ('fs_subject_id', 'subject_id')]),
                   (fs_import, brain_convert, [('brain', 'in_file')]),
                   (brain_convert, outputnode, [('out_file', 'uni_lowres')])
                   ])
    
    
    # linear registration epi median to lowres mp2rage with bbregister
    bbregister_epi = Node(fs.BBRegister(contrast_type='t2',
                                    out_fsl_file='epi2lowres.mat',
                                    out_reg_file='epi2lowres.dat',
                                    registered_file='epi2lowres.nii.gz',
                                    init='fsl',
                                    epi_mask=True
                                    ),
                    name='bbregister_epi')
    
    coreg.connect([(inputnode, bbregister_epi, [('fs_subjects_dir', 'subjects_dir'),
                                                ('fs_subject_id', 'subject_id'),
                                                ('epi_median', 'source_file')]),
                   (bbregister_epi, outputnode, [('out_fsl_file', 'epi2lowres_mat'),
                                             ('out_reg_file', 'epi2lowres_dat'),
                                             ('registered_file', 'epi2lowres')
                                             ])
                   ])
    
    
    # convert transform to itk
    itk_epi = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                           itk_transform='epi2lowres.txt'), 
                                           name='itk')
     
    coreg.connect([(brain_convert, itk_epi, [('out_file', 'reference_file')]),
                   (inputnode, itk_epi, [('epi_median', 'source_file')]),
                   (bbregister_epi, itk_epi, [('out_fsl_file', 'transform_file')]),
                   (itk_epi, outputnode, [('itk_transform', 'epi2lowres_itk')])
                   ])
    
    # transform epi to highres
    epi2lowres = Node(ants.ApplyTransforms(dimension=3,
                                            output_image='epi2lowres_lin.nii.gz',
                                            interpolation = 'BSpline',
                                            #invert_transform_flags=[True, False]
                                            ),
                       name='epi2lowres')
    
    coreg.connect([(inputnode, epi2lowres, [('uni_lowres', 'reference_image'),
                                             ('epi_median', 'input_image')]),
                   (itk_epi, epi2lowres, [('itk_transform', 'transforms')]),
                   (epi2lowres, outputnode, [('output_image', 'epi2lowres_lin')])])


    return coreg