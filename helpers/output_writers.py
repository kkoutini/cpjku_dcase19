import torch

import pandas as pd

import shared_globals
import numpy as np

def sid_dcase18t1(s):
    return s.replace("audio/", "").replace(".wav", "")

def sid_audiodataset(s):
    return s.replace("audio/", "").replace(".wav", "")


def default_csv(sid, out, trainer, dataset_name, name,
                label_encoder, id_col_name="Id", id_preprocess_func="",
                label_col_name="label", to_csv_args={}):
    """
    csv writer
    :param args: contains custom args
    :param name: predicting dataset configuration
    :param sid: sample ids
    :param out: torch tensor of model outputs
    :param trainer: trainer object
    """
    le = trainer.run.get_command_function(label_encoder)()
    _, preds = torch.max(out, dim=1)
    preds = le.inverse_transform(preds)
    if id_preprocess_func != "":
        prfunc = globals()[id_preprocess_func]
        sid = [prfunc(s) for s in sid]
    df = pd.DataFrame({id_col_name: sid, label_col_name: preds})
    df.set_index(id_col_name, inplace=True)
    file_name = trainer.config.out_dir + "/" + dataset_name + "_" + name + ".csv"
    shared_globals.console.info("prediction to: " + file_name)
    df.to_csv(file_name, **to_csv_args)

def binary_csv(sid, out, trainer, dataset_name, name,
                class_name="true",threshold=0, id_col_name="Id", id_preprocess_func="",
                label_col_name="label", to_csv_args={}):
    """
    csv writer
    :param args: contains custom args
    :param name: predicting dataset configuration
    :param sid: sample ids
    :param out: torch tensor of model outputs
    :param trainer: trainer object
    """

    p = (out > threshold).detach().cpu().numpy()
    preds= np.zeros(p.shape[0], dtype="S20").astype(str)
    preds[p]=class_name
    if id_preprocess_func != "":
        prfunc = globals()[id_preprocess_func]
        sid = [prfunc(s) for s in sid]
    df = pd.DataFrame({id_col_name: sid, label_col_name: preds})
    df.set_index(id_col_name, inplace=True)
    file_name = trainer.config.out_dir + "/" + dataset_name + "_" + name + ".csv"
    shared_globals.console.info("prediction to: " + file_name)
    df.to_csv(file_name, **to_csv_args)

def default_torch(sid, out, trainer, dataset_name, name):
    file_name = trainer.config.out_dir + "/" + dataset_name + "_" + name + ".pt"
    shared_globals.console.info("prediction to: " + file_name)
    torch.save((sid, out), file_name)
