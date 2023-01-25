# Graph Neural Network for Simulated X-Ray Transient Detection

## Description

The present work aims to train a GNN to label a particular type of X-Ray transient using simulated events overlayed onto real data from XMM-Newton observations. We will experiment with Graph Convolutional Networks (GCNs). We will therefore  have to trandsform our point-cloud data into a "k nearest neighbors"-type graph. Raw data is taken from icaro.iusspavia.it `/mnt/data/PPS_ICARO_SIM2`. Observations store data for each photon detected, with no filter applied, in FITS files ending in `EVLI0000.FTZ` for the original observations and `EVLF0000.FTZ` for the observation and simulation combined. We will refer to the former datafiles as "genuine" and to the latter as "simulated" for brevity. This package allows training for single-node single-GPU models, single-node multi-GPU models, or multi-node multi-GPU models with SLURM. These capabilities are based on a template that can be found at the following [link](https://github.com/ramyamounir/Template). This project is meant to be run on the Beluga cluster by ComputeCanada, but is versatile enough to be run elsewhere too.

## Dependencies

Once you cloned the repository, access its directory. In the virtual environment of your choosing (generated either through `conda` or `virtualenv`) execute

```bash
python install_requirements.py
```

This should install all requirements contained in [requirements.txt](requirements.txt) through `pip`, handling the non-standard installation process for `pytorch_geometric` and associated packages. Notice that, if you want to include more options to `pip` you can by simpy appending such options to the commandline. As example, if you want to execute a dry run you may add the opportune option

```bash
python install_requirements.py --dry run
```

> **Warning**
> Trying to modify the `-r | --requirements` option will result in an unhandled exception being thrown and termination of the script.

> **Warning**
> Due to the non-standard installation routine needed for `pythorch_geometric`, the `--dry-run` option will correctly work only on the other packages, unless they were already previously installed.

> **Note**
> For Beluga users, you may want to run the command with the `--no-index` option first, so that if local wheels are already available they will be employed.

## Usage

In the package directory, execute
```bash
python generate_default_config.py
```

This will generate the `config.yml` file with default values. **Do not execute the program using default values**, since they are placeholders. Modify the config file according to your needs, you may also save different config files under different names, then  execute the training routine by running

```bash
python train.py -cfg <config file name>
```

> **Note**
> You may override any setting in the config file by simply adding `-<configuration name> [<value>]` to the execution line.

Some settings, such as `dataset` and `model`, should not be changed unless you add the appropriate source code to the package, and should not be needed when simply reviewing the package.

## Layout

This package follows a modular approach where the main components of the code (architecture, loss, scheduler, trainer, etc.) are organized into subdirectories.

- The [train.py](train.py) script contains all the arguments (parsed by argparse) and nodes/GPUS initializer (slurm or local). It also contains code for importing the dataset, model, loss function and passing them to the trainer function.
- The [lib/trainer/trainer.py](lib/trainer/trainer.py) script defines the details of the training procedure.
- The [lib/dataset/[args.dataset].py](lib/datasets/) imports data and defines the dataset function. Creating a data directory with a soft link to the dataset is recommended.
- The [lib/core/](lib/core/) directory contains definitions for loss, optimizer, scheduler functions.
- The [lib/utils/](lib/utils/) directory contains helper functions organized by file name. (i.e., helper functions for distributed training are placed in the lib/util/distributed.py file).

## Run

### For single node, single GPU training:

Try the following example
```
python train.py -gpus 0
```

### For single node, multi GPU training:

Try the following example
```
python train.py -gpus 0,1,2
```

### For single node, multi GPU training on SLURM:

Try the following example
```
python train.py -slurm -slurm_nnodes 1 -slurm_ngpus 4 -slurm_partition general
```

### For multi node, multi GPU training on SLURM:

Try the following example
```
python train.py -slurm -slurm_nnodes 2 -slurm_ngpus 8  -slurm_partition general
```

## Tips

- To get more information about available arguments, run: ```python train.py -h```;
- To automatically start Tensorboard server as a different thread, add the argument: ``` -tb ```;
- To overwrite model log files and start from scratch, add the argument: ``` -reset ```; otherwise, it will use the last weights as a checkpoint and continue writing to the same Tensorboard log files - if the same model name is used;
- To choose specific node names on SLURM, use the argument: ``` -slurm_nodelist GPU17,GPU18 ``` as an example;
- If running on a GPU with Tensor cores, using mixed precision models can speed up your training. Add the argument ``` -fp16 ``` to try it out. If it makes training unstable due to the loss of precision, don't use it;
- The stdout and stderr will be printed in the out directory. We only print the first GPU output. Make sure to change the out directory in the config file depending on the cluster you are using;
- if you find a bug in this package, open a new issue or a pull request. Any collaboration is more than welcome!

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
