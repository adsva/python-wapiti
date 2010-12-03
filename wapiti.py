#!/usr/bin/env python
"""
python-wapiti provides Python bindings for libwapiti, my shared
library branch of wapiti (http://wapiti.limsi.fr), a very nice
sequence labeling tool written by Thomas Lavergne.
"""
__author__ = "Adam Svanberg <asvanberg@gmail.com>"
__version__ = "0.1"

import ctypes
import sys

_wapiti = ctypes.CDLL('libwapiti.so')

#
# Helper types
#

class FILEType(ctypes.Structure):
    """stdio.h FILE type"""
    pass
FILE_p = ctypes.POINTER(FILEType)
PyFile_AsFile = ctypes.pythonapi.PyFile_AsFile
PyFile_AsFile.restype = FILE_p
PyFile_AsFile.argtypes = [ctypes.py_object]

class TmsType(ctypes.Structure):
    """clock.h timer type"""
    _fields_ = [
        ('tms_utime', ctypes.c_long),
        ('tms_stime', ctypes.c_long),
        ('tms_cutime', ctypes.c_long),
        ('tms_cstime', ctypes.c_long)
        ]


#
# Wapiti types
#

class LbfgsType(ctypes.Structure):
    _fields_ = [
        ('clip', ctypes.c_bool),
        ('histsz', ctypes.c_int),
        ('maxls', ctypes.c_int),
        ]
class Sgdl1Type(ctypes.Structure):
    _fields_ = [
        ('eta0', ctypes.c_double),
        ('alpha', ctypes.c_double),
        ]
class BcdType(ctypes.Structure):
    _fields_ = [
        ('kappa', ctypes.c_double),
        ]
class RpropType(ctypes.Structure):
    _fields_ = [
        ('stpmin', ctypes.c_double),
        ('stpmax', ctypes.c_double),
        ('stpinc', ctypes.c_double),
        ('stpdec', ctypes.c_double),
        ]

class OptType(ctypes.Structure):
    _fields_ = [
        ('mode', ctypes.c_int),
        ('input', ctypes.c_char_p),
        ('output', ctypes.c_char_p),
        ('maxent', ctypes.c_bool),
        ('algo', ctypes.c_char_p),
        ('pattern', ctypes.c_char_p),
        ('model', ctypes.c_char_p),
        ('devel', ctypes.c_char_p),
        ('compact', ctypes.c_bool),
        ('sparse', ctypes.c_bool),
        ('nthread', ctypes.c_int),
        ('maxiter', ctypes.c_int),
        ('rho1', ctypes.c_double),
        ('rho2', ctypes.c_double),
        ('objwin', ctypes.c_int),
        ('stopwin', ctypes.c_int),
        ('stopeps', ctypes.c_double),
        ('lbfgs', LbfgsType),
        ('sgdl1', Sgdl1Type),
        ('bcd', BcdType),
        ('rprop', RpropType),
        ('label', ctypes.c_bool),
        ('check', ctypes.c_bool),
        ('outsc', ctypes.c_bool),
        ('lblpost', ctypes.c_bool),
        ('nbest', ctypes.c_int),
        ]


_default_options = OptType.in_dll(_wapiti, "opt_defaults")
def add_default_options(options):
    for field in _default_options._fields_:
        field_name = field[0]
        if not field_name in options:
            options[field_name] = getattr(_default_options, field_name)
    if options['maxiter'] == 0:
        # Zero means max int size. Handled by option parser in wapiti.
        options['maxiter'] = sys.maxsize

 
class ModelType(ctypes.Structure):
    _fields_ = [
        ('opt', ctypes.POINTER(OptType)),    # options for training

	# Size of various model parameters
        ('nlbl', ctypes.c_size_t),   # Y   number of labels       
        ('nobs', ctypes.c_size_t),   # O   number of observations 
        ('nftr', ctypes.c_size_t),   # F   number of features     

        # Information about observations
        ('kind', ctypes.c_char_p),   # [O]  observations type     
        ('uoff', ctypes.c_void_p),   # [O]  unigram weights offset
        ('boff', ctypes.c_void_p),   # [O]  bigram weights offset 

	# The model itself
        ('theta', ctypes.c_void_p),  # [F]  features weights

        # Datasets
        ('train', ctypes.c_void_p),  # training dataset    
        ('devel', ctypes.c_void_p),  # development dataset
        ('reader', ctypes.c_void_p), 

        # Stopping criterion
        ('werr', ctypes.c_void_p),   # Window of error rate of last iters
        ('wcnt', ctypes.c_int),      # Number of iters in the window     
        ('wpos', ctypes.c_int),      # Position for the next iter        

        # Timing
        ('timer', TmsType),
        ('total', ctypes.c_double)]
    

# Setup methods

# Reader
_wapiti.rdr_new.argtypes = [ctypes.c_bool]
_wapiti.rdr_new.restype = ctypes.c_void_p
# Model
_wapiti.mdl_new.argtypes = [ctypes.c_void_p]
_wapiti.mdl_new.restype = ctypes.POINTER(ModelType)

# File based train, label, dump
_wapiti.dotrain.argtypes = [ctypes.POINTER(ModelType)]
_wapiti.dolabel.argtypes = [ctypes.POINTER(ModelType)]
_wapiti.dodump.argtypes = [ctypes.POINTER(ModelType)]


class Model:
    
    def __init__(self, encoding='utf8', **options):
        self.encoding = encoding
        
        add_default_options(options) # Add any missing values
        self.options = OptType(**options)

        self._model = _wapiti.mdl_new(_wapiti.rdr_new(self.options.maxent))
        self._model.contents.opt = ctypes.pointer(self.options)

        if self.options.model:
            f = file(self.options.model)
            fp = ctypes.pythonapi.PyFile_AsFile(f)
            _wapiti.mdl_load(self._model, fp)


    # Simple file based wrappers for the three main modes of Wapiti

    def train(self, infile, outfile):
        """File-based training"""
        self.options.input = infile
        self.options.output = outfile
        _wapiti.dotrain(self._model)

    def label(self, infile, outfile):
        """File-based labeling"""
        self.options.input = infile
        self.options.output = outfile
        _wapiti.dolabel(self._model)

    def dump(self, infile, outfile):
        """File-based model dump"""
        self.options.input = infile
        self.options.output = outfile
        _wapiti.dodump(self._model)
        

    # Libwapiti extras

    def label_sequence(self, lines):
        """Returns a list of label predictions for a list of BIO-formatted lines"""
        # Redefining types seems to be one of the suggested ways of
        # dealing with variable sized arrays.  This sets the array
        # size for the input lines and the output labels.
        numlines = len(lines)

        class RawType(ctypes.Structure):
            _fields_ = [
                ('len', ctypes.c_int), 
                ('lines', ctypes.c_char_p * numlines)
            ]
            
        _wapiti.tag_label_raw.argtypes = [
            ctypes.POINTER(ModelType), 
            ctypes.POINTER(RawType), 
            (ctypes.c_char_p * (self._model.contents.opt.contents.nbest * numlines))]

        # If the input is unicode, convert to the specified encoding 
        lines = [line.encode(self.encoding) if type(line) == unicode else line for line in lines]

        # Build the char* array from the input lines
        raw_lines = RawType(numlines, (ctypes.c_char_p * numlines)(*lines))

        # Build empty label char* array based on number of lines 
        # TODO:
        # The actual labels are allocated in libwapiti, need to find
        # out if they're ever freed. Not sure if the python conversion
        # copies or not.
        labels = (ctypes.c_char_p * (self._model.contents.opt.contents.nbest * numlines))()

        _wapiti.tag_label_raw(self._model, ctypes.pointer(raw_lines), labels)

        return labels



if __name__ == '__main__':
    # Emulate wapiti functionality. Not really meaningful except for
    # testing the python bindings
    
    import optparse

    parser = optparse.OptionParser("usage: %prog train|label|dump [options] [infile] [outfile]")
    option_dict = {}
    def dictsetter(option, opt_str, value, *args, **kwargs):
        option_dict[option.dest] = value

    parser.add_option('--me', dest='maxent', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--algo', dest='algo', type='string', default='l-bfgs', 
                      action='callback', callback=dictsetter)
    parser.add_option('--pattern', dest='pattern', type='string', default=None, 
                      action='callback', callback=dictsetter)
    parser.add_option('--model', dest='model', type='string', default=None, 
                      action='callback', callback=dictsetter)
    parser.add_option('--devel', dest='devel', type='string', default=None, 
                      action='callback', callback=dictsetter)
    parser.add_option('--compact', dest='compact', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--sparse', dest='sparse', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--nthread', dest='nthread', type='int', default=1, 
                      action='callback', callback=dictsetter)
    parser.add_option('--maxiter', dest='maxiter', type='int', default=0, 
                      action='callback', callback=dictsetter)
    parser.add_option('--rho1', dest='rho1', type='float', default=0.5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--rho2', dest='rho2', type='float', default=0.0001, 
                      action='callback', callback=dictsetter)
    parser.add_option('--objwin', dest='objwin', type='int', default=5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stopwin', dest='stopwin', type='int', default=5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stopeps', dest='stopeps', type='float', default=0.02, 
                      action='callback', callback=dictsetter)
    parser.add_option('--clip', dest='clip', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--histsz', dest='clip', type='int', default=5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--maxls', dest='clip', type='int', default=40, 
                      action='callback', callback=dictsetter)
    parser.add_option('--eta0', dest='eta0', type='float', default=0.8, 
                      action='callback', callback=dictsetter)
    parser.add_option('--alpha', dest='alpha', type='float', default=0.85, 
                      action='callback', callback=dictsetter)
    parser.add_option('--kappa', dest='kappa', type='float', default=1.5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stpmin', dest='stpmin', type='float', default=1e-8, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stpmax', dest='stpmax', type='float', default=50.0, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stpinc', dest='stpinc', type='float', default=1.2, 
                      action='callback', callback=dictsetter)
    parser.add_option('--stpdec', dest='stpdec', type='float', default=0.5, 
                      action='callback', callback=dictsetter)
    parser.add_option('--label', dest='label', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--check', dest='check', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--score', dest='outsc', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--post', dest='lblpost', type='int', default=False, 
                      action='callback', callback=dictsetter)
    parser.add_option('--nbest', dest='nbest', type='int', default=1, 
                      action='callback', callback=dictsetter)

    options, args = parser.parse_args()

    action = len(args) > 0 and args[0] or None
    if not action:
        parser.error("Invalid action")

    infile = len(args) > 1 and args[1] or None
    outfile = len(args) > 2 and args[2] or None

    model = Model(**option_dict)

    if action == 'train':
        model.train(infile, outfile)
    elif action == 'label':
        if 'model' not in option_dict:
            parser.error("Labeling requires a model")
        model.label(infile, outfile)
    elif action == 'dump':
        model.dump(infile, outfile)
    else:
        parser.error("Invalid action")

        
        
    
