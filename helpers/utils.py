# coding: utf-8

import importlib
import math
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import helpers.criterion as cr

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def load_model(config, exper=None):
    module = importlib.import_module('models.{}'.format(config['arch']))
    if exper is not None:
        exper.add_source_file(
            str(os.path.dirname(os.path.realpath(__file__))) + "/../models/" + config['arch'].replace(".", "/") + ".py")
    Network = getattr(module, 'Network')
    return Network(config)


def load_strategy(strategy):
    module = importlib.import_module('strategies.{}'.format(strategy))
    do_train_epoch = getattr(module, 'do_train_epoch')
    return do_train_epoch


def save_checkpoint(state, outdir):
    model_path = os.path.join(outdir, "models", 'model_state.pth')
    best_model_path = os.path.join(outdir, "models", 'model_best_state.pth')
    torch.save(state, model_path)
    if state['best_epoch'] == state['epoch']:
        shutil.copy(model_path, best_model_path)
        shutil.copy(model_path, os.path.join(outdir, "models", "model_state_{}.pth".format(str(state['epoch']))))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, num):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] = self.sum.get(k, 0) + val[k] * num
            self.count[k] = self.count.get(k, 0) + num
            self.avg[k] = self.sum[k] / self.count[k]


class NestedDictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.dict = None

    def rec_update(self, dic, val, num):
        if isinstance(val, dict):
            if dic is None:
                dic = {}
            for k in val:
                dic[k] = self.rec_update(dic.get(k), val[k], num)
            return dic
        else:
            if dic is None:
                dic = AverageMeter()
            dic.update(val, num)
            return dic

    def update(self, val, num):
        self.dict = self.rec_update(self.dict, val, num)

    def rec_sum(self, val):
        if isinstance(val, dict):
            res = {}
            for k in val:
                res[k] = self.rec_sum(val[k])
            return res
        else:
            return val.sum

    def get_sum(self):
        return self.rec_sum(self.dict)

    def rec_avg(self, val):
        if isinstance(val, dict):
            res = {}
            for k in val:
                res[k] = self.rec_avg(val[k])
            return res
        else:
            return val.avg

    def get_avg(self):
        return self.rec_avg(self.dict)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def _get_optimizer(model_parameters, optim_config):
    if optim_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=optim_config['base_lr'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config['nesterov'])
    elif optim_config['optimizer'] == 'adabound':
        optimizer = AdaBound(
            model_parameters,
            lr=optim_config['base_lr'],
            final_lr=optim_config['final_lr'],
            gamma=optim_config['gamma'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay'],
            amsbound=optim_config.get("amsgrad") or False)
    elif optim_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optim_config['base_lr'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay'], amsgrad=optim_config.get("amsgrad") or False)
    return optimizer


def _get_scheduler(optimizer, optim_config):
    if optim_config['optimizer'] == 'sgd':
        if optim_config['scheduler'] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optim_config['milestones'],
                gamma=optim_config['lr_decay'])
        elif optim_config['scheduler'] == 'cosine':
            total_steps = optim_config['epochs'] * \
                          optim_config['steps_per_epoch']

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    total_steps,
                    1,  # since lr_lambda computes multiplicative factor
                    optim_config['lr_min'] / optim_config['base_lr']))
        elif optim_config['scheduler'] == "swa":
            scheduler = SWA_schedule(optimizer, optim_config['scheduler_args'])
        elif optim_config['scheduler'] == "mycos":
            scheduler = Cos_schedule(optimizer, optim_config['scheduler_args'])
        elif optim_config['scheduler'] == 'linear':
            scheduler = Linear_schedule(optimizer, optim_config['scheduler_args'])
        elif optim_config['scheduler'] == 'drop':
            scheduler = Drop_schedule(optimizer, optim_config['scheduler_args'])
        else:
            scheduler = None
    else:
        if optim_config['scheduler'] == 'linear':
            scheduler = Linear_schedule(optimizer, optim_config['scheduler_args'])
        elif optim_config['scheduler'] == 'drop':
            scheduler = Drop_schedule(optimizer, optim_config['scheduler_args'])
        elif optim_config['scheduler'] == "mycos":
            scheduler = Cos_schedule(optimizer, optim_config['scheduler_args'])
        else:
            scheduler = None
    return scheduler


class Cos_schedule:
    def __init__(self, optimizer, args):
        self.lr_init = args['lr_init']
        self.epochs = args['epochs']
        self.optimizer = optimizer
        self.start = args.get("start", 40)
        self.lr = self.lr_init
        self.lr_low = args.get('lr_low') or 0.

    def step(self, epoch):
        if epoch < self.start:
            return
        self.lr = self.lr_low + 0.5 * (self.lr_init - self.lr_low) * (math.cos(math.pi * epoch / self.epochs) + 1)
        adjust_learning_rate(self.optimizer, self.lr)

    def get_lr(self):
        return [self.lr]


class Linear_schedule:
    def __init__(self, optimizer, args):
        self.lr_init = args['lr_init']
        self.epochs = args['epochs']
        self.start = args.get("start", 40)
        self.warmup = args.get("warmup", 0) or 0
        self.lr_warmup = args.get("lr_warmup", None) or (self.lr_init / 100)
        self.after_lr = args.get("after_lr", self.lr_init / 100)
        self.optimizer = optimizer
        self.lr = self.lr_init

    def step(self, epoch):
        if epoch < self.warmup:
            self.lr = self.lr_warmup + (self.lr_init - self.lr_warmup) * epoch / self.warmup
        elif epoch < self.start:
            self.lr = self.lr_init
        elif epoch >= self.epochs:
            self.lr = self.after_lr
        else:
            self.lr = self.lr_init - self.lr_init * (epoch - self.start) / (self.epochs - self.start)
        adjust_learning_rate(self.optimizer, self.lr)

    def get_lr(self):
        return [self.lr]


class Drop_schedule:
    def __init__(self, optimizer, args):
        self.lr_init = args['lr_init']
        self.drop_list = args.get("drop_list", [])
        self.warmup = args.get("warmup", 0) or 0
        self.lr_warmup = args.get("lr_warmup", None) or (self.lr_init / 100)
        self.optimizer = optimizer
        self.lr = self.lr_init
        self.current_list_index = 0

    def step(self, epoch):
        if epoch <= self.warmup:
            self.lr = self.lr_warmup + (self.lr_init - self.lr_warmup) * epoch / self.warmup
            adjust_learning_rate(self.optimizer, self.lr)
        if self.current_list_index < len(self.drop_list):
            # self.drop_list <> epoch,factor,reset_optimizer
            e = self.drop_list[self.current_list_index][0]
            f = self.drop_list[self.current_list_index][1]
            rest_optimizer_beta = self.drop_list[self.current_list_index][2]

            if epoch == e:
                self.lr *= f
                self.current_list_index += 1
                adjust_learning_rate(self.optimizer, self.lr)
                if rest_optimizer_beta:
                    print("\n\n\nSETING BETA TO ", rest_optimizer_beta, "\n\n\n")
                    for param_group in self.optimizer.param_groups:
                        param_group['betas'] = (rest_optimizer_beta, 0.999)  # Only change beta1

            else:
                # nothing to do
                return

    def get_lr(self):
        return [self.lr]


class SWA_schedule:
    def __init__(self, optimizer, args):
        self.lr_init = args['lr_init']
        self.swa_start = args['swa_start']
        self.swa_lr = args['swa_lr']
        self.optimizer = optimizer
        self.lr = self.lr_init

    def step(self, epoch):
        t = (epoch) / (self.swa_start)
        lr_ratio = self.swa_lr / self.lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        self.lr = self.lr_init * factor
        adjust_learning_rate(self.optimizer, self.lr)

    def get_lr(self):
        return [self.lr]


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def my_mixup(data, targets, alpha, mode=None):
    rn_indices = torch.randperm(data.size(0))
    lambd = np.random.beta(alpha, alpha, data.size(0))
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd.astype(np.float32))
    # data = data * lam + data2 * (1 - lam)
    # targets = targets * lam + targets2 * (1 - lam)
    return rn_indices, lam


def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)


criterion_names = sorted(name for name in cr.__dict__
                         if not name.startswith("__")
                         and name.startswith("criterion_")
                         and callable(cr.__dict__[name]))

eval_names = []

total_eval_names = []

event_listeners_names =[]


def get_criterion(name):
    if 'criterion_' + name not in criterion_names:
        print('criterion_' + name, " not in ", str(criterion_names))
        print("check for typos or wether is it implemented!")
    assert 'criterion_' + name in criterion_names, 'criterion_' + name + " is not in " + str(criterion_names)
    return cr.__dict__['criterion_' + name]


def get_evaluation(name):
    assert 'eval_' + name in eval_names, 'eval_' + name + " is not in " + str(eval_names)
    return evl.__dict__['eval_' + name]


def get_total_evaluation(name):
    assert 'total_eval_' + name in total_eval_names, 'total_eval_' + name + " is not in " + str(eval_names)
    return evl.__dict__['total_eval_' + name]


def get_event_listener(name):
    assert 'event_listener_' + name in event_listeners_names, 'total_eval_' + name + " is not in " + str(
        event_listeners_names)
    return ev_lster.__dict__['event_listener_' + name]


tttext = ""


def apply_linefeeds(text):
    """
    Interpret backspaces and linefeeds in text like a terminal would.

    Interpret text like a terminal by removing backspace and linefeed
    characters and applying them line by line.

    If final line ends with a carriage it keeps it to be concatenable with next
    output chunk.
    """

    orig_lines = text.split('\n')
    orig_lines_len = len(orig_lines)
    new_lines = []
    for orig_line_idx, orig_line in enumerate(orig_lines):
        orig_line_len = len(orig_line)
        if orig_line.startswith("\x1b[2K"):
            continue
        k = orig_line.rfind("\r")
        if k == -1:
            new_lines.append(orig_line)
        else:
            tmps = orig_line[k:]
            if len(tmps):
                new_lines.append(tmps)

    return '\n'.join(new_lines)


## swa
def swa_moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    # print(momenta.values())
    n = 0
    for input, _, _ in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b
        print('\x1b[2K' + 'swa_recal momentom={}'.format(
            momentum), end="\r")

    model.apply(lambda module: _set_momenta(module, momenta))

    # for input, _, _ in loader:
    #     input = input.cuda(async=True)
    #     input_var = torch.autograd.Variable(input)
    #     print('\x1b[2K' + 'swa_recal 2 momentom={}'.format(
    #         momentum), end="\r")
    #     model(input_var)


def count_parameters(model, trainable=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    Example Usage:
    >>> def f(x):
    ...     print 'f(%s)' % x
    >>> def g(x):
    ...     print 'g(%s)' % x
    >>> e = Event()
    >>> e()
    >>> e.append(f)
    >>> e(123)
    f(123)
    >>> e.remove(f)
    >>> e()
    >>> e += (f, g)
    >>> e(10)
    f(10)
    g(10)
    >>> del e[0]
    >>> e(2)
    g(2)

    """

    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)


def compare_dictionaries(dict_1, dict_2, dict_1_name="d1", dict_2_name="d2", path=""):
    """Compare two dictionaries recursively to find non mathcing elements

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns:

    """
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + ".%s" % k
        if not k in dict_2:
            key_err += "Key %s%s not in %s\n" % (dict_2_name, path, dict_2_name)
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(dict_1[k], dict_2[k], 'd1', 'd2', path)
            else:
                if dict_1[k] != dict_2[k]:
                    value_err += "Value of %s%s (%s) not same as %s%s (%s)\n" \
                                 % (dict_1_name, path, dict_1[k], dict_2_name, path, dict_2[k])

    for k in dict_2.keys():
        path = old_path + ".%s" % k
        if not k in dict_1:
            key_err += "Key %s%s not in %s\n" % (dict_2_name, path, dict_1_name)

    return key_err + value_err + err


def worker_init_fn(x):
    seed = (torch.initial_seed() + x * 1000) % 2 ** 31  # problem with nearly seeded randoms

    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    return
