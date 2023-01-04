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

import os
from glob import glob

import numpy as np
import torch

import pyg_lib #new in torch_geometric 2.2.0!
from torch_geometric.data import Data
from torch_geometric.data import Dataset

import torch_geometric.transforms as ttr

class SimTransientData(Data):
    """
    Subclass of `torch_geometric.data.Data` that adds a `pos` property to store and retrieve the position data. The position data is stored in the last three values of the `x` attribute.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    pos : np.ndarray
        3-D position data.
    """
    def __init__(self, x = None, edge_index = None, edge_attr = None, y = None, pos = None, **kwargs):
        assert pos is None, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
                             " values of the `x` attribute. Please append the position coordinates to your 'x' parameter")
        assert x.shape[1] >= 3, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
                                 " values of the `x` attribute. Therefore the 'x' parameter must contain at least three elements.")
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    @property
    def pos(self):
        """
        Getter for the position data.
        
        Returns
        -------
        np.ndarray
            3-D position data.
        """
        return self.x[:, -3:]

    @pos.setter
    def pos(self, replace):
        """
        Setter for the position data.
        
        Parameters
        ----------
        replace : np.ndarray
            3-D position data to be set.
        
        Raises
        ------
        AssertionError
            If the shape of `replace` is not (num_points, 3).
        """
        assert replace.shape == self.pos.shape
        self.x[:, -3:] = replace


class SimTransientDataset(Dataset):
    """
    A dataset for loading and interacting with data stored in PyTorch files.

    Args:
        root (str): The root directory of the dataset.
        pattern (str, optional): A glob pattern to match file names. Defaults to '*EVLF000.FTZ*'.
        transform (callable, optional): A function/transform to apply to the data. Defaults to None.
    """
    def __init__(self, root, pattern='*EVLF000.FTZ*', transform=None):
        super().__init__(root=root, transform=transform)
        self._raw_dir = root
        search_path = os.path.join(root, pattern)
        self.filenames = sorted(glob.glob(search_path))
        self.file_count = len(self.filenames)

    @property
    def raw_dir(self):
        """str: The root directory of the raw data."""
        return self._raw_dir

    @property
    def processed_dir(self):
        """str: The root directory of the processed data."""
        return self._raw_dir

    @property
    def raw_file_names(self):
        """List[str]: The names of the raw files in the dataset."""
        return self.filenames

    @property
    def processed_file_names(self):
        """List[str]: The names of the processed files in the dataset."""
        return self.filenames

    def __len__(self):
        """int: The number of files in the dataset."""
        return self.file_count

    def get(self, idx):
        """
        Get a data item from the dataset by its index.

        Args:
            idx (int): The index of the data item.

        Returns:
            object: The data item at the given index.
        """
        data = torch.load(self.filenames[idx])
        return data


def get_dataset(args):

    # === Get Dataset === #
    train_dataset = SimTransientDataset(root=args.processed_data,
                                        pattern=args.processed_pattern)

    return train_dataset
