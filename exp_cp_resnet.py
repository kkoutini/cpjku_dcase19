from __future__ import print_function

import argparse
import datetime
import os
import shutil
import time
import random
import json
from trainer import Trainer

parser = argparse.ArgumentParser(description='CP_ResNet Training')
# Optimization options
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')

# Optimization options
parser.add_argument('--rho', default=5, type=int,
                    help='rho value as explained in DCASE2019 workshop paper '
                         '"Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification"')
# Optimization options
parser.add_argument('--mixup', default=1, type=int,
                    help='use mixup if 1. ')

args = parser.parse_args()

with open("configs/cp_resnet.json", "r") as text_file:
    default_conf = json.load(text_file)

default_conf['out_dir'] = default_conf['out_dir'] + str(datetime.datetime.now().strftime('%b%d_%H.%M.%S'))

print("The experiment outputs will be found at: ", default_conf['out_dir'])
tensorboard_write_path = default_conf['out_dir'].replace("out", "runs", 1)
print("The experiment tesnorboard can be accessed: tensorboard --logdir  ", tensorboard_write_path)

print("Rho value : ", args.rho)
print("Use Mix-up : ", args.mixup)

from models.cp_resnet import get_model_based_on_rho

default_conf['model_config'] = get_model_based_on_rho(args.rho, config_only=True)

if args.mixup:
    default_conf['use_mixup'] = True
    default_conf['loss_criterion'] = 'mixup_default'
else:
    default_conf['use_mixup'] = False

epochs = args.epochs
trainer = Trainer(default_conf)
trainer.fit(epochs)
trainer.predict("last")
trainer.load_best_model()
trainer.predict()
