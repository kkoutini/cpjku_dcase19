import logging
import types
import functools

outdir = "./out"
dataset = None
bare_model = None

base_learning_rate = None
current_learning_rate = None
global_run_unique_identifier = ""

current_learning_rate = 0

batch_size = 0

state = None

config = None

logger = logging.getLogger("")
console = None

global_step = 0

global_epoch = 0

optim_class_id = 3

criterion = None

online_scaler = None


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


_named_counter = dict()


def named_counter_inc(name):
    _named_counter[name] = _named_counter.get(name, 0) + 1


def named_counter_get(name):
    return _named_counter.get(name, 0)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'