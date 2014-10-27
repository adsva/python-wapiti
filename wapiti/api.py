#!/usr/bin/env python
"""
python-wapiti provides Python bindings for libwapiti, my shared
library version of wapiti (http://wapiti.limsi.fr), a very nice
sequence labeling tool written by Thomas Lavergne.
"""
__author__ = "Adam Svanberg <asvanberg@gmail.com>"
__version__ = "0.1"

import re
import os
import six
import ctypes
import logging
import multiprocessing
from ctypes.util import find_library


_libwapiti_root_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '../',
)
_wapiti = ctypes.CDLL(
    os.path.join(
        _libwapiti_root_path,
        list(filter(
            lambda x: re.search('^libwapiti.*?\.so$', x),
            os.listdir(_libwapiti_root_path)
        )).pop()
    )
)
_libc = ctypes.CDLL(find_library('c'))

#
# Helper types
#


class FILEType(ctypes.Structure):
    """stdio.h FILE type"""
    pass


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
        ('histsz', ctypes.c_uint32),
        ('maxls', ctypes.c_uint32),
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
        ('cutoff', ctypes.c_bool),
    ]


class OptType(ctypes.Structure):
    _fields_ = [
        ('mode', ctypes.c_int),
        ('input', ctypes.c_char_p),
        ('output', ctypes.c_char_p),
        ('maxent', ctypes.c_bool),
        ('type', ctypes.c_char_p),
        ('algo', ctypes.c_char_p),
        ('pattern', ctypes.c_char_p),
        ('model', ctypes.c_char_p),
        ('devel', ctypes.c_char_p),
        ('rstate', ctypes.c_char_p),
        ('sstate', ctypes.c_char_p),
        ('compact', ctypes.c_bool),
        ('sparse', ctypes.c_bool),
        ('nthread', ctypes.c_uint32),
        ('jobsize', ctypes.c_uint32),
        ('maxiter', ctypes.c_uint32),
        ('rho1', ctypes.c_double),
        ('rho2', ctypes.c_double),
        ('objwin', ctypes.c_uint32),
        ('stopwin', ctypes.c_uint32),
        ('stopeps', ctypes.c_double),
        ('lbfgs', LbfgsType),
        ('sgdl1', Sgdl1Type),
        ('bcd', BcdType),
        ('rprop', RpropType),
        ('label', ctypes.c_bool),
        ('check', ctypes.c_bool),
        ('outsc', ctypes.c_bool),
        ('lblpost', ctypes.c_bool),
        ('nbest', ctypes.c_uint32),
        ('force', ctypes.c_bool),
    ]


_default_options = OptType.in_dll(_wapiti, "opt_defaults")

ALGORITHMS = [
    'l-bfgs',
    'sgd-l1',
    'bcd',
    'rprop',
    'rprop+',
    'rprop-',
    'auto',
]


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

# Model
_wapiti.api_new_model.argtypes = [ctypes.POINTER(OptType), ctypes.c_char_p]
_wapiti.api_new_model.restype = ctypes.POINTER(ModelType)
_wapiti.api_load_model.argtypes = [ctypes.c_char_p, ctypes.POINTER(OptType)]
_wapiti.api_load_model.restype = ctypes.POINTER(ModelType)


# Training
_wapiti.api_add_train_seq.argtypes = [
    ctypes.POINTER(ModelType), ctypes.c_char_p]

# Labeling
#
# The restype is set to POINTER(c_char) instead of c_char_p so we can
# handle the conversion to python string and make sure the c-allocated
# data is freed.
_wapiti.api_label_seq.argtypes = [
    ctypes.POINTER(ModelType), ctypes.c_char_p, ctypes.c_bool]
_wapiti.api_label_seq.restype = ctypes.POINTER(ctypes.c_char)

_libc.free.argtypes = [ctypes.c_void_p]

# Setup logging
#
# The logging functions in api_logs are replaced by standard python
# logging callbacks. Since the error log is supposed to be fatal, we
# wrap that logger to also raise SystemExit.


class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logger = logging.getLogger('wapiti')
logger.addHandler(NullHandler())  # Avoids warnings in non-logging applications


def fatal(msg):
    logger.error("%s - exiting", msg)
    raise SystemExit(msg)

#Define the callback (retval, *args)
LogFunc = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
#The array of 4 function pointers
api_logs = (LogFunc * 4).in_dll(_wapiti, "api_logs")
api_logs[0] = LogFunc(fatal)                         # api_logs[FATAL]
api_logs[1] = LogFunc(fatal)                         # api_logs[PFATAL]
api_logs[2] = LogFunc(logger.warning)                # api_logs[WARNING]
api_logs[3] = LogFunc(logger.info)                   # api_logs[INFO]


class Model:
    r"""
    Wapiti model wrapper that handles training and labeling

    The model takes BIO-formatted strings as input for both training
    and labeling. Since wapiti treats all whitespace as column
    delimiters, any whitespace in the token being labeled or the label
    must be escaped, or weird and not so wonderful things will happen.

    The patterns used to extract features are passed in as a string
    with one pattern per line. See http://crfpp.sourceforge.net/#templ
    for an explanation of the pattern format. There are also good
    examples in the data directory of wapiti.

    Example:
    >>> m = Model(patterns='*\nU:Wrd X=%x[0,0]')
    >>> text = 'We PRPsaw VBD\nthe DT\nlittle JJ\nyellow JJ\ndog NN'
    >>> m.add_training_sequence(text)
    >>> m.train()
    >>> m.label_sequence('We\saw\nthe\nlittle\nyellow\ndog')
    'We\\saw\tVBD\nthe\tDT\nlittle\tJJ\nyellow\tJJ\ndog\tNN\n'
    """

    def __init__(self, patterns=None, encoding='utf8', **options):

        # Make sure encoding is taken care of when passing strings
        self.encoding = encoding
        if 'nthread' not in options:
            # If thread count isn't given, use number of processors
            options['nthread'] = multiprocessing.cpu_count()

        # Add unspecified options values from Wapiti's default struct
        for field in _default_options._fields_:
            field_name = field[0]
            if not field_name in options:
                options[field_name] = getattr(_default_options, field_name)
        if options['maxiter'] == 0:
            # Wapiti specifies that 0 means max int size for this option.
            options['maxiter'] = 2147483647
        #python3 support
        for field_name, field_value in options.items():
            if isinstance(field_value, six.text_type):
                options[field_name] = field_value.encode(self.encoding)
        self.options = OptType(**options)

        if patterns:
            patterns = patterns.encode(self.encoding)

        # Load model from file if specified
        if self.options.model:
            self._model = _wapiti.api_load_model(
                self.options.model,
                ctypes.pointer(self.options)
            )
        else:
            if self.options.pattern:
                with open(self.options.pattern) as fp:
                    self.patterns = fp.read()
            else:
                self.patterns = patterns

            self._model = _wapiti.api_new_model(
                ctypes.pointer(self.options),
                self.patterns
            )

    def __del__(self):
        if _wapiti and self._model:
            _wapiti.api_free_model(self._model)

    def add_training_sequence(self, sequence):
        """
        Add a string of BIO-formatted lines to the training set
        """
        if isinstance(sequence, six.text_type):
            sequence = sequence.encode(self.encoding)
        _wapiti.api_add_train_seq(self._model, sequence)

    def train(self, sequences=None):
        if sequences:
            for seq in sequences:
                if isinstance(seq, six.text_type):
                    seq = seq.encode(self.encoding)
                _wapiti.api_add_train_seq(self._model, seq)
        _wapiti.api_train(self._model)

    def save(self, filename):
        if isinstance(filename, six.text_type):
            _wapiti.api_save_model(self._model, filename.encode("utf8"))
        else:
            raise TypeError("The file name must be unicode type")

    def label_sequence(self, lines, include_input=False):
        """
        Accepts a string of BIO-formatted lines and returns a column
        of labels.

        The input string is labeled as one sequence, i.e. double
        linebreaks are ignored.

        If include_input is True, the label column is appended to the
        input columns.
        """
        if isinstance(lines, six.text_type):
            lines = lines.encode(self.encoding)
        cp = _wapiti.api_label_seq(self._model, lines, include_input)

        # Convert to python string and free the c-allocated data
        labeled = ctypes.string_at(cp)
        _libc.free(cp)

        return labeled
