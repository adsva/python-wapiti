#!/usr/bin/env python
"""
python-wapiti provides Python bindings for libwapiti, my shared
library version of wapiti (http://wapiti.limsi.fr), a very nice
sequence labeling tool written by Thomas Lavergne.
"""
__author__ = "Adam Svanberg <asvanberg@gmail.com>"
__version__ = "0.1"

from ctypes.util import find_library
import ctypes

_wapiti = ctypes.CDLL(find_library('wapiti'))
_libc = ctypes.CDLL(find_library('c'))

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
        ('jobsize', ctypes.c_int),
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
    
                
def freeing_char_p(char_p):
    """
    Restype to copy a c_char_p to str and free memory

    Avoids memory leaks when returning char pointers allocated in C by
    converting a copy to a python string and freeing the original
    pointer.
    """
    s = ctypes.string_at(char_p)
    _libc.free(char_p)
    return s
        
# Setup methods

# Model
_wapiti.api_new_model.argtypes = [ctypes.POINTER(OptType), ctypes.c_char_p]
_wapiti.api_new_model.restype = ctypes.POINTER(ModelType)
_wapiti.api_load_model.argtypes = [ctypes.c_char_p, ctypes.POINTER(OptType)]
_wapiti.api_load_model.restype = ctypes.POINTER(ModelType)


# Training
_wapiti.api_add_train_seq.argtypes = [ctypes.POINTER(ModelType), ctypes.c_char_p]

# Labeling
_wapiti.api_label_seq.argtypes = [ctypes.POINTER(ModelType), ctypes.c_char_p]
_wapiti.api_label_seq.restype = freeing_char_p

_libc.free.argtypes = [ctypes.c_void_p]


class Model:
    
    def __init__(self, patterns=None, encoding='utf8', **options):

        # Make sure encoding is taken care of when passing strings
        self.encoding = encoding
        ctypes.set_conversion_mode(encoding, 'replace')
        
        # Add unspecified options values from Wapiti's default struct
        for field in _default_options._fields_:
            field_name = field[0]
            if not field_name in options:
                options[field_name] = getattr(_default_options, field_name)
        if options['maxiter'] == 0:
            # Wapiti specifies that 0 means max int size for this option.
            options['maxiter'] = sys.maxsize

        self.options = OptType(**options)

        # Load model from file if specified
        if self.options.model:
            self._model = _wapiti.api_load_model(
                self.options.model,
                ctypes.pointer(self.options)
            )
        else:
            if self.options.pattern:
                self.patterns = open(self.options.pattern).read()
            else:
                self.patterns = patterns

            self._model = _wapiti.api_new_model(
                ctypes.pointer(self.options),
                self.patterns
            )            

    def add_training_sequence(self, sequence):
        _wapiti.api_add_train_seq(self._model, seq)
        
    def train(self, sequences=[]):
        for seq in sequences:
            _wapiti.api_add_train_seq(self._model, seq)
        _wapiti.api_train(self._model)
        

    def save(self, modelfile):
        if type(modelfile) != file:
            modelfile = open(modelfile, 'w')
        fp = ctypes.pythonapi.PyFile_AsFile(modelfile)
        _wapiti.api_save_model(self._model, fp)
    
    def label_sequence(self, lines):
        """
        Accepts a string of BIO-formatted lines and adds a label column

        The input string is labeled as one sequence, i.e. double
        linebreaks are ignored.
        """

        if type(lines) == unicode:
            lines.encode(self.encoding)
        labeled = _wapiti.api_label_seq(self._model, lines)

        return labeled 


if __name__ == '__main__':
    # Emulate wapiti functionality. Not really meaningful except for
    # testing the python bindings
    
    import sys
    import optparse

    parser = optparse.OptionParser(
        "usage: %prog train|label [options] [infile] [outfile]"
    )

    option_dict = {}
    def dictsetter(option, opt_str, value, *args, **kwargs):
        option_dict[option.dest] = value

    parser.add_option('--me', dest='maxent', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--algo', dest='algo', type='string',  
                      action='callback', callback=dictsetter)
    parser.add_option('--pattern', dest='pattern', type='string',  
                      action='callback', callback=dictsetter)
    parser.add_option('--model', dest='model', type='string',  
                      action='callback', callback=dictsetter)
    parser.add_option('--devel', dest='devel', type='string',  
                      action='callback', callback=dictsetter)
    parser.add_option('--compact', dest='compact', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--sparse', dest='sparse', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--nthread', dest='nthread', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--jobsize', dest='jobsize', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--maxiter', dest='maxiter', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--rho1', dest='rho1', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--rho2', dest='rho2', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--objwin', dest='objwin', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stopwin', dest='stopwin', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stopeps', dest='stopeps', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--clip', dest='clip', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--histsz', dest='clip', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--maxls', dest='clip', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--eta0', dest='eta0', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--alpha', dest='alpha', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--kappa', dest='kappa', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stpmin', dest='stpmin', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stpmax', dest='stpmax', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stpinc', dest='stpinc', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--stpdec', dest='stpdec', type='float',  
                      action='callback', callback=dictsetter)
    parser.add_option('--label', dest='label', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--check', dest='check', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--score', dest='outsc', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--post', dest='lblpost', type='int',  
                      action='callback', callback=dictsetter)
    parser.add_option('--nbest', dest='nbest', type='int',  
                      action='callback', callback=dictsetter)

    options, args = parser.parse_args()

    action = len(args) > 0 and args[0] or None
    if not action:
        parser.error("Invalid action")

    infile = len(args) > 1 and open(args[1]) or sys.stdin
    outfile = len(args) > 2 and open(args[2], 'w') or sys.stdout

    model = Model(**option_dict)

    if action == 'train':
        model.train(infile.read().split('\n\n'))
        model.save(outfile)
            
    elif action == 'label':
        if 'model' not in option_dict:
            parser.error("Labeling requires a model")
        if not infile:
            infile = sys.stdin
        for sequence in infile.read().split('\n\n'):
            outfile.write(model.label_sequence(sequence))
            outfile.close()
    else:
        parser.error("Invalid action")

        
        
    
