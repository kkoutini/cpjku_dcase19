import soundfile as sf
import librosa
from sklearn import preprocessing

# All parser functions should start with "parser", define custom parsers in this file, and refernece them from config

import pandas as pd

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def owriter_default_csv(csv_file, files_col, labels_col, header=None, delimiter="\t", label_encoder=None):
    """
    csv metadata parser, reads file name and labels from CSV

    :param csv_file: csv_file
    :param files_col: the column with file names
    :param labels_col: the column with the ile labels
    :return: files, labels, label_encoder_decoder
    """

    raise NotImplementedError


def owriter_multilabel_csv(output, filename, classes, delm="\t", classes_delm=",", threshold=0.5):
    """
    csv metadata parser, reads file name and labels from CSV where multilabels are represented
    by repeating the file name in the csv (DCASE2017 task4  style)
    example:
    file1 label1
    file1 label2
    :param csv_file: csv_file
    :param files_col: the column with file names
    :param labels_col: the column with the ile labels
    :return: 2d array containing the two column files, labels
    """

    return filename + delm + classes_delm.join(classes[np.where(output > threshold)])


def owriter_multilabel_strong_prob_csv(output, filename, classes, delm="\t", classes_delm=" ", threshold=0.5):
    """
    csv metadata parser, reads file name and labels from CSV where multilabels are represented
    by repeating the file name in the csv (DCASE2017 task4  style)
    example:
    file1 label1
    file1 label2
    :param csv_file: csv_file
    :param files_col: the column with file names
    :param labels_col: the column with the ile labels
    :return: 2d array containing the two column files, labels
    """
    res = ""
    for i, c in enumerate(classes):
        res += filename + delm + c + delm + classes_delm.join([str(kk) for kk in output[i]]) + "\n"

    return res[0:-1]
