import hashlib
import os
import time
from os.path import expanduser
import numpy as np
from attrdict import AttrDefault
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import *


def parser_categorical_csv(csv_file, files_col, labels_col, header=None, delimiter="\t", label_encoder=None):
    """
    csv metadata parser, reads file name and labels from CSV

    :param csv_file: csv_file
    :param files_col: the column with file names
    :param labels_col: the column with the ile labels
    :return: files, labels, label_encoder_decoder
    """

    df = pd.read_csv(csv_file, header=header, delimiter=delimiter)
    df = df[[df.columns[files_col], df.columns[labels_col]]]
    df.columns = [0, 1]
    le = label_encoder
    if le is None:
        le = preprocessing.LabelEncoder()
        y_labels = le.fit_transform(df.iloc[:, 1])
    else:
        y_labels = le.fit_transform(df.iloc[:, 1])
    return df.iloc[:, 0].values, y_labels, le

def parser_nolabel_csv(csv_file, files_col, header=None, delimiter="\t"):
    """
    csv metadata parser, reads file name and labels from CSV, for evaluation sets

    :param csv_file: csv_file
    :param files_col: the column with file names
    :param labels_col: the column with the ile labels
    :return: files, labels, label_encoder_decoder
    """

    df = pd.read_csv(csv_file, header=header, delimiter=delimiter)
    df = df[[df.columns[files_col]]]
    df.columns = [0]

    y_labels = np.zeros(len(df.iloc[:].values), dtype=np.long)
    return df.iloc[:, 0].values, y_labels, None


def get_meta_parser(parser):
    return parser_categorical_csv



df_trset = None

loaded_dataset = None
label_encoder = None
loaded_dataset_sub = None

class DatasetsManager:
    def __init__(self, config):
        self.config = AttrDefault(lambda: None, config)

    def get_dataset(self, config):
        return getattr(self, config.dataset.split(".")[1])()

    def get_sub_dataset(self):
        normalize = self.config['normalize']
        if not normalize:
            return self.df_get_sub_dataset()
        # make sure norms are loaded
        self.get_test_set()
        print("normalized SUB dataset!")
        ds = self.df_get_sub_dataset()
        return PreprocessDataset(ds, norm_func)

    def df_get_sub_dataset(self):
        name, normalize, audio_path, parser, parser_args, audio_processor, cache, cache_x_name, file_cache = \
            self.config['name'], self.config['normalize'], self.config['audio_path'], self.config['parser'], \
            self.config[
                'parser_args'], self.config['audio_processor'], self.config['cache'], self.config['cache_x_name'], \
            self.config['file_cache']
        sub_audio_path, sub_parser, sub_parser_args = self.config['sub_audio_path'],self.config['sub_parser'],\
                                                      self.config['sub_parser_args']
        sub_audio_path = os.path.expanduser(sub_audio_path)
        global loaded_dataset_sub
        global label_encoder
        if loaded_dataset_sub is not None:
            return loaded_dataset_sub

        print("loading dataset from '{}'".format(name + "_sub"))

        def getdatset():
            meta_parser = parser_nolabel_csv
            global label_encoder
            files, labels, _ = meta_parser(**sub_parser_args)

            return AudioPreprocessDataset(files, labels, label_encoder, sub_audio_path, audio_processor)

        if cache:
            if file_cache:
                loaded_dataset_sub = FilesCachedDataset(getdatset, name + "_sub",
                                                        x_name=cache_x_name + "_sub_ap_" + audio_processor)
        else:
            loaded_dataset_sub = getdatset()
        return loaded_dataset_sub

    def get_full_dataset(self):
        name, normalize, audio_path, parser, parser_args, audio_processor, cache, cache_x_name, file_cache = \
            self.config['name'], self.config['normalize'], self.config['audio_path'], self.config['parser'], \
            self.config[
                'parser_args'], self.config['audio_processor'], self.config['cache'], self.config['cache_x_name'], \
            self.config['file_cache']

        audio_path = os.path.expanduser(audio_path)
        global loaded_dataset
        global label_encoder
        if loaded_dataset is not None:
            return loaded_dataset
        print("loading dataset from '{}'".format(name))
        if normalize:
            print("normalizing dataset")

        def getdatset():
            meta_parser = get_meta_parser(parser)
            global label_encoder
            files, labels, label_encoder = meta_parser(**parser_args)

            return AudioPreprocessDataset(files, labels, label_encoder, audio_path, audio_processor)

        if cache and file_cache:
            loaded_dataset = FilesCachedDataset(getdatset, name, x_name=cache_x_name + "_ap_" + audio_processor)

        else:
            loaded_dataset = getdatset()
        return loaded_dataset

    def df_get_train_set(self):

        train_files_csv, fold = self.config.train_files_csv, self.config.fold
        global df_trset
        if df_trset is not None:
            return df_trset
        df_trset = SelectionDataset(self.get_full_dataset(),
                                    pd.read_csv(train_files_csv.format(fold), header=0, sep="\t")['filename'].values)

        return df_trset

    def df_get_test_set(self):
        test_files_csv, fold = self.config.test_files_csv, self.config.fold
        totalset = self.get_full_dataset()

        return SelectionDataset(totalset,
                                pd.read_csv(test_files_csv.format(fold), header=0, sep="\t")['filename'].values)

    def get_test_set(self):
        normalize = self.config.normalize
        if not normalize:
            return self.df_get_test_set()
        print("normalized test!")
        name, fold, train_files_csv, audio_processor = self.config.name, \
                                                       self.config.fold, self.config.train_files_csv, \
                                                       self.config.audio_processor
        fill_norms(name, fold, train_files_csv, audio_processor, self.df_get_train_set())
        ds = self.df_get_test_set()
        return PreprocessDataset(ds, norm_func)

    def get_train_set(self):
        normalize, subsample, roll, \
        stereo_desync, stereo_flip, \
        vertical_desync, spec_resize = self.config.normalize, \
                                       self.config.subsample, self.config.roll, self.config.stereo_desync, \
                                       self.config.stereo_flip, self.config.vertical_desync, self.config.spec_resize

        name, fold, train_files_csv, audio_processor = self.config.name, \
                                                       self.config.fold, self.config.train_files_csv, \
                                                       self.config.audio_processor
        ds = self.df_get_train_set()

        if subsample == 1:
            print("subsample train!")
            ds = PreprocessDataset(ds, subsample_func)
        elif subsample == 2:
            print("subsample  train to 10 secs!")
            ds = PreprocessDataset(ds, subsample_func2)
        if normalize:
            print("normalized train!")
            fill_norms(name, fold, train_files_csv, audio_processor, self.df_get_train_set())
            ds = PreprocessDataset(ds, norm_func)
        if roll:
            ds = PreprocessDataset(ds, get_roll_func())
        if stereo_flip:
            ds = PreprocessDataset(ds, get_stereo_flip_func())
        if stereo_desync:
            ds = PreprocessDataset(ds, get_desync_stereo_roll_func())
        if vertical_desync:
            ds = PreprocessDataset(ds, get_vertical_desync_func())

        return ds


def get_roll_func(axis=2, shift=None):
    def roll_func(x):
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-50, 50))
        global FirstTime

        return x.roll(sf, axis)

    return roll_func


def get_desync_stereo_roll_func(shift=None):
    def roll_func(x):
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-1, 1))
        global FirstTime
        y = x[0]
        y = y.roll(sf, 1)
        x[0] = y
        return x

    return roll_func


def get_vertical_desync_func(shift=None):
    def roll_func(x):
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-1, 1))
        global FirstTime

        x = x.roll(sf, 1)

        return x

    return roll_func


def get_stereo_flip_func(audio_processor):
    if audio_processor == "k19_stereo":
        def flip_func(x):
            sf = int(np.random.random_integers(0, 1))
            global FirstTime
            if sf:
                y = x[0]
                x[0] = x[1]
                x[1] = y
                y = x[2]
                x[2] = x[3]
                x[3] = y
            return x

        return flip_func

    def flip_func(x):
        sf = int(np.random.random_integers(0, 1))
        global FirstTime
        if sf:
            y = x[0]
            x[0] = x[1]
            x[1] = y
        return x

    return flip_func


class FilesCachedDataset(Dataset):
    def __init__(self, get_dataset_func, dataset_name, x_name="", y_name="",
                 cache_path="datasets/cached_datasets/",
                 ):
        self.dataset = None

        def getDataset():
            if self.dataset == None:
                self.dataset = get_dataset_func()
            return self.dataset

        self.get_dataset_func = getDataset
        self.x_name = x_name + y_name
        cache_path = expanduser(cache_path)
        self.cache_path = os.path.join(cache_path, dataset_name, "files_cache", self.x_name)
        try:
            original_umask = os.umask(0)
            os.makedirs(self.cache_path, exist_ok=True)
        finally:
            os.umask(original_umask)

    def __getitem__(self, index):
        cpath = os.path.join(self.cache_path, str(index) + ".pt")
        try:
            return torch.load(cpath)
        except FileNotFoundError:
            tup = self.get_dataset_func()[index]
            torch.save(tup, cpath)
            return tup

    def get_ordered_labels(self):
        return self.get_dataset_func().get_ordered_labels()

    def get_ordered_ids(self):
        return self.get_dataset_func().get_ordered_ids()

    def get_xcache_path(self):
        return os.path.join(self.cache_path, self.x_name + "_x.pt")

    def get_ycache_path(self):
        return os.path.join(self.cache_path, self.y_name + "_y.pt")

    def get_sidcache_path(self):
        return os.path.join(self.cache_path, self.y_name + "_sid.pt")

    def __len__(self):
        return len(self.get_dataset_func())


class PreprocessDataset(Dataset):
    """A bases preprocessing dataset representing a preprocessing step of a Dataset preprossessed on the fly.


    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        if callable(preprocessor):
            self.preprocessor = preprocessor
        else:
            print("preprocessor is not calllable ", preprocessor)

    def __getitem__(self, index):
        x, id, y = self.dataset[index]
        return self.preprocessor(x), id, y

    def get_ordered_ids(self):
        return self.dataset.get_ordered_ids()

    def get_ordered_labels(self):
        return self.dataset.get_ordered_labels()

    def __len__(self):
        return len(self.dataset)


class SelectionDataset(Dataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.


        supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, sample_ids):
        self.available_indexes = []
        self.dataset = dataset
        self.reselect(sample_ids)

    def reselect(self, sample_ids):
        reverse_dict = dict([(sid, i) for i, sid in enumerate(self.dataset.get_ordered_ids())])
        self.available_indexes = [reverse_dict[sid] for sid in sample_ids]

    def get_ordered_ids(self):
        return self.sample_ids

    def get_ordered_labels(self):
        raise NotImplementedError("Maybe reconsider caching only a selection Dataset. why not select after cache?")

    def __getitem__(self, index):
        return self.dataset[self.available_indexes[index]]

    def __len__(self):
        return len(self.available_indexes)


class AudioPreprocessDataset(Dataset):
    """A bases preprocessing dataset representing a Dataset of files that are loaded and preprossessed on the fly.

    Access elements via __getitem__ to return: preprocessor(x),sample_id,label

    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, files, labels, label_encoder, base_dir, preprocessor, return_tensor=True):
        self.files = files
        self.labels = labels
        self.label_encoder = label_encoder
        self.base_dir = base_dir
        self.preprocessor = get_audio_processor(preprocessor)
        self.return_tensor = return_tensor

    def __getitem__(self, index):
        x = self.preprocessor(self.base_dir + self.files[index])
        if self.return_tensor and not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x, self.files[index], self.labels[index]

    def get_ordered_ids(self):
        return self.files

    def get_ordered_labels(self):
        return self.labels

    def __len__(self):
        return len(self.files)


import audioprocessors as ap

processors_names = sorted(name for name in ap.__dict__
                          if name.islower() and not name.startswith("__")
                          and name.startswith("processor_")
                          and callable(ap.__dict__[name]))


def get_audio_processor(name):
    if 'processor_' + name not in processors_names:
        print('processor_', name, " not in ", str(processors_names))
        print("check for typos or is it implemented!")
    assert 'processor_' + name in processors_names
    return ap.__dict__['processor_' + name]


class ObjectCacher:
    def __init__(self, get_obj_func, dataset_name, obj_name="",
                 cache_path="datasets/cached_datasets/", verbose=True):
        self.dataset_name = dataset_name
        self.obj_name = obj_name
        cache_path = expanduser(cache_path)
        self.cache_path = os.path.join(cache_path, dataset_name)
        try:
            startTime = time.time()
            xpath = self.get_obj_cache_path()

            if verbose:
                print(
                    "attempting to load x from cache at " + xpath + "...")
            self.obj = torch.load(xpath)

            if verbose:
                endTime = time.time()
                print(
                    "loaded " + xpath + " from cache in %s " % (endTime - startTime))
        except IOError:
            if verbose:
                print(
                    "Invalid cache " + xpath + " , recomputing")
            self.obj = get_obj_func()
            saveStartTime = time.time()
            torch.save(self.obj, xpath)
            if verbose:
                endTime = time.time()
                print(
                    "loaded " + obj_name + " in %s, and cached in %s, total %s seconds " % (
                        (saveStartTime - startTime),
                        (endTime - saveStartTime), (endTime - startTime)))

    def get_obj_cache_path(self):
        return os.path.join(self.cache_path, self.obj_name + "_obj.pt")

    def get(self):
        return self.obj


def norm_func(x):
    return (x - tr_mean) / tr_std


def subsample_func(x):
    k = torch.randint(x.size(2) - x.size(1) + 1, (1,))[0].item()
    return x[:, :, k:k + x.size(1)]


def subsample_func2(x):
    k = torch.randint(x.size(2) - 431 + 1, (1,))[0].item()
    return x[:, :, k:k + 431]


def cal_mean_single_thread():
    trainset = df_trainset
    c = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c.append(x)
    t = torch.stack(c).transpose(2, 3).contiguous()
    print(t.size())
    t = t.view(-1, t.size()[3])
    print(t.size())
    m = t.mean(0).float().reshape(1, 256, 1)
    print("mean", m.size())
    return m


def cal_mean():
    trainset = df_trainset
    c = []
    print("cal_mean ")
    lengsths_sum = 0
    for i, (x, _, _) in enumerate(torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8)):
        # = trainset[i]
        if i == 0:
            print(x.shape)
        lengsths_sum += x.shape[3]
        x = x[0]
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        c.append(x)
    print("average length", lengsths_sum / len(trainset))
    print("c [0,1]= ", c[0].size(), c[1].size())
    t = torch.cat(c)  # .transpose(2, 3).contiguous()
    print(t.size())
    m = t.mean(0).float().reshape(1, c[0].size(1), 1)
    print("mean", m.size())
    del t
    return m


def cal_mean_with_deltas():
    trainset = df_trainset
    c1 = []
    c2 = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c1.append(x[0:2])
        c2.append(x[2:])
    t1 = torch.stack(c1).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m0 = t1.mean(0).float().reshape(256, 1)
    print("mean", m0.size())
    t1 = torch.stack(c2).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m1 = t1.mean(0).float().reshape(256, 1)
    print("mean", m1.size())
    m = torch.stack([m0, m0, m1])
    print("all mean", m.size())

    return m


def cal_std_with_deltas():
    trainset = df_trainset
    c1 = []
    c2 = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c1.append(x[0:2])
        c2.append(x[2:])
    t1 = torch.stack(c1).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m0 = t1.std(0).float().reshape(256, 1)
    print("std", m0.size())
    t1 = torch.stack(c2).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m1 = t1.std(0).float().reshape(256, 1)
    print("std", m1.size())
    m = torch.stack([m0, m0, m1])
    print("all std", m.size())

    return m


def cal_mean_per_2channel():
    trainset = df_trainset
    c1 = []
    c2 = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c1.append(x[0:2])
        c2.append(x[2:])
    t1 = torch.stack(c1).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m0 = t1.mean(0).float().reshape(256, 1)
    print("mean", m0.size())
    t1 = torch.stack(c2).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m1 = t1.mean(0).float().reshape(256, 1)
    print("mean", m1.size())
    m = torch.stack([m0, m0, m1, m1])
    print("all mean", m.size())

    return m


def cal_std_per_2channel():
    trainset = df_trainset
    c1 = []
    c2 = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c1.append(x[0:2])
        c2.append(x[2:])
    t1 = torch.stack(c1).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m0 = t1.std(0).float().reshape(256, 1)
    print("std", m0.size())
    t1 = torch.stack(c2).transpose(2, 3).contiguous()
    print(t1.size())
    t1 = t1.view(-1, t1.size()[3])
    print(t1.size())
    m1 = t1.std(0).float().reshape(256, 1)
    print("std", m1.size())
    m = torch.stack([m0, m0, m1, m1])
    print("all std", m.size())
    return m


def cal_std_single_thread():
    trainset = df_trainset
    c = []
    for i in range(len(trainset)):
        x, _, _ = trainset[i]
        c.append(x)
    t = torch.stack(c).transpose(2, 3).contiguous()
    print(t.size())
    t = t.view(-1, t.size()[3])
    print(t.size())
    sd = t.std(0).float().reshape(1, 256, 1)
    print("sd", sd.size())
    return sd


def cal_std():
    trainset = df_trainset
    c = []
    for i, (x, _, _) in enumerate(torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=10)):
        # x, _, _ = trainset[i]
        x = x[0]
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        c.append(x)
    print("c [0,1]= ", c[0].size(), c[1].size())
    t = torch.cat(c)  # .transpose(2, 3).contiguous()
    print(t.size())
    sd = t.std(0).float().reshape(1, c[0].size(1), 1)
    print("sd", sd.size())
    return sd


df_trainset = None
tr_mean = None
tr_std = None


def fill_norms(name, fold, train_files_csv, audio_processor, training_set):
    global tr_mean, tr_std, df_trainset
    df_trainset = training_set
    mnfunc = cal_mean
    if audio_processor == "k19_stereo":
        print("k19_stereo mean calc")
        mnfunc = cal_mean_per_2channel
    if audio_processor == "m18_stereo_deltas":
        print("deltas mean calc")
        mnfunc = cal_mean_with_deltas
    if tr_mean is None:
        tr_mean = ObjectCacher(mnfunc, name,
                               "tr_mean_{}_f{}_ap{}".format(h6(train_files_csv), fold, audio_processor)).get()
        # print("train_val:", torch.all(torch.eq(tr_mean, tr_mean_val)))
    stdfunc = cal_std
    if audio_processor == "k19_stereo":
        print("k19_stereo std calc")
        stdfunc = cal_std_per_2channel
    if audio_processor == "m18_stereo_deltas":
        print("deltas std calc")
        stdfunc = cal_std_with_deltas
    if tr_std is None:
        tr_std = ObjectCacher(stdfunc, name,
                              "tr_std_{}_f{}_ap{}".format(h6(train_files_csv), fold, audio_processor)).get()
        # print("sd_val:", torch.all(torch.eq(tr_mean, tr_mean_val)))


def h6(w):
    return hashlib.md5(w.encode('utf-8')).hexdigest()[:6]
