from nibabel.analyze import AnalyzeImage, AnalyzeHeader
from nibabel.spm99analyze import Spm99AnalyzeHeader, Spm99AnalyzeImage
from nibabel.spatialimages import HeaderTypeError

from nibabel.volumeutils import (pretty_mapping, endian_codes, native_code,
                          swapped_code, make_dt_codes)
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import guessed_image_type as nibabel_guess
from nibabel import spm99analyze
import os
import numpy as np
from nibabel.arrayproxy import ArrayProxy
from nibabel.fileholders import FileHolder


# Sub-parts of standard 4dfp/ifh header from
# Washington School of Medicine
acquisition_key_dtd = [
    ('version_of_keys', 'S3'),
    ('image_modality', 'S3'),
    ('originating_system', 'S20'),
    ('conversion_program', 'S10'),
    ('program_version', 'S30'),
    ('original_institution', 'S30'),
    ]
data_key_dtd = [
    ('name_of_data_file', 'S80'),
    ('date', 'S11'),
    ('patient_ID', 'S35'),
    ]

voxel_size_dtd = [
    ('scaling_factor_(mm/pixel)_[1]', 'f4'),
    ('scaling_factor_(mm/pixel)_[2]', 'f4'),
    ('scaling_factor_(mm/pixel)_[3]', 'f4'),
    ('slice_thickness_(mm/pixel)', 'f4'),
    ('center', 'S50'),
    ('mmppix', 'S50'),
                  ]
image_size_dtd = [
    ('matrix_size_[1]', 'i2'),
    ('matrix_size_[2]', 'i2'),
    ('matrix_size_[3]', 'i2'),
    ('matrix_size_[4]', 'i2'),
                  ]
image_dimension_dtd = [
    ('time_series_flag', 'i2'),
    ('number_of_dimensions', 'i2'),
    ('global_maximum', 'i4'),
    ('global_minimum', 'i4'),
    ('fwhm_in_voxels', 'f4'),
    ]
data_dtype_dtd = [
    ('number_format', 'S10'),
    ('number_of_bytes_per_pixel', 'i2'),
    ('orientation', 'i2'), 
    ('imagedata_byte_order', 'S10'),
                  ]
mri_info_dtd = [
    ('mri_parameter_file_name', 'S80'),
    ('mri_sequence_file_name', 'S30'),
    ('mri_sequence_description', 'S10'),
    ]
analyze_hdr = spm99analyze.header_key_dtd + \
                spm99analyze.data_history_dtd + \
                spm99analyze.image_dimension_dtd

# Full header numpy dtype combined across sub-fields
header_dtype = np.dtype(acquisition_key_dtd + \
                        data_key_dtd + \
                        voxel_size_dtd + \
                        image_size_dtd + \
                        image_dimension_dtd + \
                        mri_info_dtd + \
                        data_dtype_dtd + \
                        analyze_hdr)

_dtdefs = ( # code, conversion function, equivalent dtype, aliases
    (0, 'none', np.void),
    (1, 'binary', np.void), # 1 bit per voxel, needs thought
    (2, 'uint8', np.uint8),
    (4, 'int16', np.int16),
    (8, 'int32', np.int32),
    (16, 'float32', np.float32),
    #(16, 'float32', np.dtype('>f4')),
    (32, 'complex64', np.complex64), # numpy complex format?
    (64, 'float64', np.float64),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')])),
    (255, 'all', np.void))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)


class WashUHeader(Spm99AnalyzeHeader):
        
    # Copies of module-level definitions
    template_dtype = header_dtype
    _data_type_codes = data_type_codes
    # fields with recoders for their values
    _field_recoders = {'datatype': data_type_codes}
    # default x flip
    default_x_flip = True

    # data scaling capabilities
    has_data_slope = False
    has_data_intercept = False
    
    
    def __init__(self, header_dict=None, 
                 endianness=None, 
                 check=True):
                
        Spm99AnalyzeHeader.__init__(self, 
                                    endianness=endianness,
                                    check=check)
        print header_dict
        self._general = None
        self._data_type = None
        self._header_dict = header_dict
        if isinstance(self._header_dict, dict):
            self.set_header_info()
        
        
    @classmethod
    def default_structarr(klass, endianness=None):
        return super(klass, klass).default_structarr(endianness=endianness)
    
    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        
        if fileobj.mode == 'rb':
            fileobj = FileHolder(fileobj._name).get_prepare_fileobj(mode='r')
        
        header_dict = klass.file_to_dict(fileobj.fobj)
        endianness = klass.guessed_endian(header_dict)
        return klass(header_dict=header_dict, endianness=endianness)
        
    @classmethod
    def file_to_dict(klass, fileobj, separator='\t:= '):
        
        b = fileobj.readline()
        if b.find('INTERFILE') == -1:
            raise HeaderTypeError('Header File not supported! Check filename.')
        
        header_dict = dict()
        
        while b != '':
            b = fileobj.readline()
            try:
                key_, value_ = b.split(separator)
            except ValueError:
                break                
            key_ = clean_key(key_)
            value_ = parse_key_value(key_, value_, header_dtype)
            header_dict[str(key_)] = value_
        
        fileobj.close()

        return header_dict
    
    @classmethod
    def guessed_endian(klass, hdr):
        
        if 'imagedata_byte_order' not in hdr.keys():
            endianess = '>'
        elif hdr['imagedata_byte_order'] == 'bigendian\n':
            endianess = '>'
        else:
            endianess = '<'
        return endianess
    
    def copy(self):
        print self.__class__
        hdr_copy = self.__class__(header_dict=self._header_dict, endianness=self.endianness)
        return hdr_copy
    
    def set_header_info(self):
        
        hdr = self
        header_dict = hdr._header_dict

        #Getting image dimensions
        dim = header_dict['number_of_dimensions']

        shape = []
        for i in range(dim):
            key_ = 'matrix_size_['+str(i+1)+']'
            shape.append(header_dict[key_])
        
        hdr.set_data_shape(tuple(shape))
        
        #Get image voxel size
        zooms = []
        if 'mmppix' in header_dict.keys():
            zooms = np.abs(np.fromstring(header_dict['mmppix'], sep=' '))
            zooms = np.append(zooms, [1.])
        else:
            #hdr['orient'] = self.get_flip_orientation(header_dict['orientation'])
            for i in range(3):
                key_ = 'scaling_factor_(mm/pixel)_['+str(i+1)+']'
                zooms.append(header_dict[key_])
                zooms.append(1)

        hdr.set_zooms(tuple(zooms))
          
        if 'center' in header_dict.keys():
            origin = np.fromstring(header_dict['center'], sep=' ')
            origin = np.append(origin, [0., 0.])
            hdr['origin'] = origin
        
        #Get pixel coding information
        if header_dict['imagedata_byte_order'] == 'bigendian\n':
            endianess = '>'
        else:
            endianess = '<'
        
        if header_dict['number_format'] == 'float\n':
            format_ = 'f'
        else:
            format_ = 'i'
            
        bytes_ = str(header_dict['number_of_bytes_per_pixel'])
        
        data_type = endianess + format_ + bytes_
        
        #hdr.set_data_dtype(data_type)
        
        return

        

class WashUImage(Spm99AnalyzeImage):
    
    files_types = (('image','.img'), 
                   ('header','.ifh'),
                   ('mat', '.mat'))
    header_class = WashUHeader
    ImageArrayProxy = ArrayProxy
    
    def __init__(self, dataobj, affine, header=None,
                 extra=None, file_map=None):
        
        Spm99AnalyzeImage.__init__(self, dataobj, affine, header, extra, file_map)
        
    @classmethod
    def from_filename(klass, filename):

        return super(klass, klass).from_filename(filename)
    
#######################################################################################


def load(filename, **kwargs):

    return guessed_image_type(filename).from_filename(filename, **kwargs)

def guessed_image_type(filename):
    
    froot, ext, trailing = splitext_addext(filename, ('.gz', '.bz2'))
    lext = ext.lower()
    
    if lext == '.img' and os.path.isfile(os.path.join(froot,'.hdr')):
        return nibabel_guess(filename)
    else:
        
        return WashUImage
    
    


def clean_key(key_):
        
        return key_.replace(' ','_').split('__')[0]
    
    
def parse_key_value(key_, value_, dtype_):
        
        if key_ in dtype_.fields.keys():
            return np.array(value_, dtype=dtype_[key_])
        else:
            return np.array(value_, dtype=np.void)
    
    