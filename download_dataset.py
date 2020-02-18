

import dcase_util


import argparse
import os
import shutil
import time
import random
from trainer import  Trainer


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--version', default="2019")
args = parser.parse_args()


print("use the arg --version [2016:2019] to work with older DCASE datasets. ")
datasets_map={
    "2016": "TUTAcousticScenes_2016_DevelopmentSet",
    "2017": "TUTAcousticScenes_2017_DevelopmentSet",
    "2018": "TUTUrbanAcousticScenes_2018_DevelopmentSet",
    "2019": "TAUUrbanAcousticScenes_2019_DevelopmentSet",
    "2019eval": "TAUUrbanAcousticScenes_2019_EvaluationSet",

}
print("chosen dataset: ",datasets_map[args.version])

# from task1 baseline at http://dcase.community/challenge2019/

ds_path="./datasets/"
dcase_util.utils.Path().create(
    paths=ds_path
)
dcase_util.datasets.dataset_factory(
    dataset_class_name=datasets_map[args.version],
    data_path=ds_path,
).initialize().log()
