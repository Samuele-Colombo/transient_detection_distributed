# This code is a modified version of code originally licensed under the Apache 2.0 license.
#
# The original code can be found at https://github.com/ramyamounir/Template.
#
# Changes made by Samuele Colombo are Copyright 2022 Samuele Colombo.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from lib.utils.file import bool_flag
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder
from lib.utils.flatten import flatten_dict

import submitit, random, sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('--cfg', type=str, help='Configuration file')

    # PATHS
    paths_group = parser.add_argument_group("PATHS")
    paths_group.add_argument('--data', type=str, help='Path to raw dataset directory')
    paths_group.add_argument('--processed_data', type=str, help='Path to processed dataset directory')
    paths_group.add_argument('--processed_pattern', type=str, help='Pattern for processed data files')
    paths_group.add_argument('--genuine_pattern', type=str, help='Pattern for genuine data files')
    paths_group.add_argument('--simulated_pattern', type=str, help='Pattern for simulated data files')
    paths_group.add_argument('--out', type=str, help='Path to out directory')

    # GENERAL
    general_group = parser.add_argument_group("GENERAL")
    general_group.add_argument('--model', type=str, help='Model name')
    general_group.add_argument('--reset', action='store_true', help='Reset saved model logs and weights')
    general_group.add_argument('--tb', action='store_true', help='Start TensorBoard')
    general_group.add_argument('--gpus', type=str, help='GPUs list, only works if not on slurm')
    general_group.add_argument('--k_neighbors', type=int, help='Number of neighbors to consider in k-NN algorithm')

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument('--num_layers', type=int, help='Number of layers')
    model_group.add_argument('--hidden_dim', type=int, help='Number of nodes in the hidden layer.')

    # Dataset
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument('--dataset', type=str, help='Dataset to choose')
    dataset_group.add_argument('--batch_per_gpu', type=int, help='Batch size per gpu')
    dataset_group.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    dataset_group.add_argument('--workers', type=int, help='Number of workers')
    dataset_group.add_argument('--keys', type=str, nargs='+', help='Columns to be used as data features')

    # Architecture
    architecture_group = parser.add_argument_group("Architecture")
    architecture_group.add_argument('--arch', type=str, help='Architecture to choose')

    # Trainer
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument('--trainer', type=str, help='Trainer to choose')
    trainer_group.add_argument('--epochs', type=int, help='Number of epochs')
    trainer_group.add_argument('--save_every', type=int, help='Save model every n epochs')
    trainer_group.add_argument('--fp16', action='store_true', help='Use fp16')

    # Optimization
    optimization_group = parser.add_argument_group("Optimization")
    optimization_group.add_argument('--optimizer', type=str, help='Optimizer to choose between "adam", "sgd", and "adagrad"')
    optimization_group.add_argument('--lr_start', type=float, help='Learning rate start')
    optimization_group.add_argument('--lr_end', type=float, help='Learning rate end')
    optimization_group.add_argument('--lr_warmup', type=int, help='Learning rate warmup')

    # SLURM
    slurm_group = parser.add_argument_group("SLURM")
    slurm_group.add_argument('--slurm', action='store_true', help='Use slurm')
    slurm_group.add_argument('--slurm_ngpus', type=int, help='Number of gpus per node')
    slurm_group.add_argument('--slurm_nnodes', type=int, help='Number of nodes')
    slurm_group.add_argument('--slurm_nodelist', type=str, help='Node list')
    slurm_group.add_argument('--slurm_partition', type=str, help='Partition')
    slurm_group.add_argument('--slurm_timeout', type=int, help='Timeout')


    args = parser.parse_args()

    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            import ruamel.yaml as yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in flatten_dict(yml):
            if k not in cmd:
                args.__dict__[k] = v

    return args


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args)


def main():

    args = parse_args()
    args.port = random.randint(49152,65535)
    
    if args.slurm:

        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.output_dir = get_shared_folder() / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=12*args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.model)
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")


    else:
        init_dist_node(args)
        mp.spawn(train, args = (args,), nprocs = args.ngpus_per_node)
	

def train(gpu, args):

    # === SET ENV === #
    init_dist_gpu(gpu, args)
    
    # === DATA === #
    get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset), fromlist=["get_dataset"]), "get_dataset")
    dataset = get_dataset(args)

    sampler = DistributedSampler(dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    loader = DataLoader(dataset=dataset, 
                        sampler = sampler,
                        batch_size=args.batch_per_gpu, 
                        num_workers= args.workers,
                        pin_memory = True,
                        drop_last = True
                       )
    print(f"Data loaded")

    # === MODEL === #
    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    setattr(args, "input_dim", dataset.num_node_features)
    setattr(args, "outpup_dim", dataset.num_classes)
    setattr(args, "activation_function", nn.functional.relu)
    model = get_model(args).cuda(args.gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # === LOSS === #
    setattr(args, "loader", loader)
    from lib.core.loss import get_loss
    loss = get_loss(args).cuda(args.gpu)

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    optimizer = get_optimizer(model, args)

    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer), fromlist=["Trainer"]), "Trainer")
    Trainer(args, loader, model, loss, optimizer).fit()


if __name__ == "__main__":
    main()
