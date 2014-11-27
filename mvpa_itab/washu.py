from nibabel.analyze import AnalyzeImage, AnalyzeHeader
from nibabel.spatialimages import HeaderTypeError
from nibabel.volumeutils import make_dt_codes
import numpy as np

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
    ('mmpix', 'S50'),
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

# Full header numpy dtype combined across sub-fields
header_dtype = np.dtype(acquisition_key_dtd + \
                        data_key_dtd + \
                        voxel_size_dtd + \
                        image_size_dtd + \
                        image_dimension_dtd + \
                        mri_info_dtd + \
                        data_dtype_dtd)

_dtdefs = ( # code, conversion function, equivalent dtype, aliases
    (0, 'none', np.void),
    (1, 'binary', np.void), # 1 bit per voxel, needs thought
    (2, 'uint8', np.uint8),
    (4, 'int16', np.int16),
    (8, 'int32', np.int32),
    (16, 'float32', np.float32),
    (32, 'complex64', np.complex64), # numpy complex format?
    (64, 'float64', np.float64),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')])),
    (255, 'all', np.void))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)


class WashUHeader(AnalyzeHeader):
        
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
    
    
    def __init__(self):
        
        self._header_dict = None
        self._general = None
        self._data_type = None
        AnalyzeHeader.__init__(self)
        
    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        
        hdr = klass()
        hdr._header_dict = hdr.file_to_dict(fileobj)
        
    
    def file_to_dict(self, fileobj, separator=':= '):
        
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
            key_ = self.clean_key(key_)
            value_ = self.parse_key_value(key_, value_, self.template_dtype)
            header_dict[str(key_)] = value_
        
        return header_dict
    
            
    def clean_key(self, key_):
        
        return key_.replace(' ','_').split('__')[0]
    
    
    def parse_key_value(self, key_, value_, dtype_):
        
        data_shape = []
        if key_ in dict(image_size_dtd).keys():
            data_shape.append(np.array(value_, dtype=dtype_[key_]))
        elif key_ in dict(data_dtype_dtd).keys():
            if self._data_type == None:
                continue
        else:
            return np.array(value_, dtype=np.void)
    
    
    def build_data_dtype(self, k_, v_):
        
        if k_ == 'number_format' and v_ ==:
            
            
        
        
    
class WashUImage(AnalyzeImage):
    
    def __init__(self):
        
        AnalyzeImage.__init__(self, data, affine, header, extra, file_map)
        
        
        

def load_4dfp(filename, **kwargs):
    
    return klass
    
    