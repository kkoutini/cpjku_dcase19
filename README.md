# CP JKU Submission for DCASE 2019




## Requirements

[Conda]( https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda ) should be installed on the system.

```install_dependencies.sh``` installs the following:
* Python 3.6.3
* PyTorch  
* torchvision
* [tensorboard-pytorch]( https://github.com/lanpa/tensorboard-pytorch )
* etc..

## Installation
* Install [Anaconda](https://www.anaconda.com/) or conda

* Run the install dependencies script:
```bash
./install_dependencies.sh
```
This creates conda environment ```cpjku_dcase19``` with all the dependencies.

Running
``` source activate cpjku_dcase19``` is needed before running ```exp*.py```


## Usage
After installing dependencies:

- Activate Conda environment created by ```./install_dependencies.sh```
    ```bash
    $ source activate cpjku_dcase19
    ```

- Download the dataset:
    ```bash
    $ python download_dataset.py --version 2019
    ```
    You can also download previous versions of DCASE ```--version year```, year is one of 2018,2017,2016,2019.
    
    Alternatively, if you already have the dataset downloaded:
    - You can make link to the dataset: 
    ```bash
    ln -s ~/some_shared_folder/TAU-urban-acoustic-scenes-2019-development ./datasets/TAU-urban-acoustic-scenes-2019-development
    ```
    
    - Change the paths in ```config/[expermient_name].json```.
    
- Run the experiment script:
    ```
    $ CUDA_VISIBLE_DEVICES=0 python exp_[expeirment_name].py 
    ```
- The output of each run is stored in ``outdir``, you can also monitor the experiments with TensorBoard, using the logs stored in the tensorboard runs dir ```runsdir```. 
 Example: 
     ```bash
     tensorboard --logdir   ./runsdir/cp_resnet/exp_Aug20_14.11.28
     ```
 The exact commmand is printed when you run the experiment script.

## Example runs
### DCASE 2019 DCASE 1A
#### CP_ResNet
default adapted receptive field RN1,RN1 (in Koutini2019Receptive below):
```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py 
```
Large receptive Field
```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py  --rho 15
```
very small max receptive Field:

```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py  --rho 2
```


# Missing Features
This repo is used to publish for our submission to DCASE 2019 and MediaEval 2019. If some feauture/architecture/dataset missing feel free to contact the authors or to open an issue.

# Citation

If use this repo please cite   [The Receptive Field as a Regularizer ]( https://arxiv.org/abs/1907.01803 ) ,  Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification
:

```
@INPROCEEDINGS{Koutini2019Receptive,
AUTHOR={ Koutini, Khaled and Eghbal-zadeh, Hamid and Dorfer, Matthias and Widmer, Gerhard},
TITLE={{The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification}},
booktitle = {Proceedings of the European Signal Processing Conference (EUSIPCO)},
ADDRESS={A Coru\~{n}a, Spain},
YEAR=2019
}


@inproceedings{KoutiniDCASE2019CNNVars,
  title = {Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification},
  booktitle = {Preprint},
  date = {2019-10},
  author = {Koutini, Khaled and Eghbal-zadeh, Hamid and Widmer, Gerhard},
}

 ```