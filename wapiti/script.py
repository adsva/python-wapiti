#encoding:utf-8
import sys
import logging
import optparse
from .api import logger, Model


def parse_command():
    parser = optparse.OptionParser(
        "usage: %prog train|label|test [options] [infile] [outfile]"
    )

    option_dict = {}

    def dictsetter(option, opt_str, value, *args, **kwargs):
        option_dict[option.dest] = value

    parser.add_option('--me', dest='maxent', type='int',
                      action='callback', callback=dictsetter)
    parser.add_option('--type', dest='type', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--algo', dest='algo', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--pattern', dest='pattern', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--model', dest='model', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--devel', dest='devel', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--rstate', dest='rstate', type='string',
                      action='callback', callback=dictsetter)
    parser.add_option('--sstate', dest='sstate', type='string',
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
    parser.add_option('--force', dest='force', type='int',
                      action='callback', callback=dictsetter)
    return option_dict, parser


def run_script():
    # Emulate wapiti functionality. Not really meaningful except for
    # testing the python bindings

    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    option_dict, parser = parse_command()
    _, args = parser.parse_args()

    action = len(args) > 0 and args[0] or None
    if not action:
        parser.error("Invalid action")

    infile = len(args) > 1 and open(args[1]) or sys.stdin
    outfile = len(args) > 2 and open(args[2], 'w') or sys.stdout

    model = Model(**option_dict)

    if action == 'train':
        model.train(infile.read().strip().split('\n\n'))
        model.save(outfile)

    elif action == 'label':
        if 'model' not in option_dict:
            parser.error("Labeling requires a model")
        if not infile:
            infile = sys.stdin
        for sequence in infile.read().split('\n\n'):
            outfile.write(model.label_sequence(sequence)+'\n')
        outfile.close()
    elif action == 'test':
        import doctest
        doctest.testmod()
    else:
        parser.error("Invalid action")
