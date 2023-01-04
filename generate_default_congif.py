# Copyright 2022 Samuele Colombo.
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

def main():
  data = """PATHS:
    data: '/path/to/raw/data'  # Path to raw dataset directory
    processed_data: '/path/to/processed/data'  # Path to processed dataset directory
    processed_pattern: '*EVLF0000.FTZ.pt'  # Pattern for processed data files
    genuine_pattern: '*EVLI0000.FTZ'  # Pattern for genuine data files
    simulated_pattern: '*EVLF0000.FTZ'  # Pattern for simulated data files
    out: 'out'  # Path to out directory
  GENERAL:
    reset: false  # Reset saved model logs and weights
    tb: true  # Start TensorBoard
    gpus: 0  # GPUs list, only works if not on slurm
    k_neighbors: 6  # Number of neighbors to consider in k-NN algorithm
  Model:
    model: 'gcn'  # Model name
    num_layers: 2  # Number of layers
    hidden_dim: 4  # Number of nodes in the hidden layer.
  Dataset:
    dataset: 'sim_transient_dataset'  # Dataset to choose
    batch_per_gpu: 96  # Batch size per gpu
    shuffle: true  # Shuffle dataset
    workers: 2  # Number of workers
    keys:
    - these
    - are
    - the
    - keys  # Columns to be used as data features
  Architecture:
    arch: 'mlp'  # Architecture to choose
  Trainer:
    trainer: 'trainer'  # Trainer to choose
    epochs: 1000  # Number of epochs
    save_every: 10  # Save model every n epochs
    fp16: true  # Use fp16
  Optimization:
    optimizer: 'adam'  # Optimizer to choose between 'adam', 'sgd', and 'adagrad'
    lr_start: 0.0005  # Learning rate start
    lr_end: 1e-06  # Learning rate end
    lr_warmup: 10  # Learning rate warmup
  SLURM:
    slurm: true  # Use slurm
    slurm_ngpus: 8  # Number of gpus per node
    slurm_nnodes: 2  # Number of nodes
    # slurm_nodelist: # Node list, comment if not specified
    slurm_partition: 'general' # Partition
    slurm_timeout: 2800 # Timeout
  """

  # Open a file in write mode
  with open('config.yml', 'w') as f:
    f.write(data)

if __name__ == "__main__":
  main()