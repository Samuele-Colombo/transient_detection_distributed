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

import numpy as np

def total_len(dataset):
    """
    Returns the number of target rows of the dataset.
    
    Parameters
    ----------
    dataset : list
        A list of datasets.
    
    Returns
    -------
    int
        The total number of target rows in the dataset.
    """
    return np.sum([len(data.y) for data in dataset])

def total_positives(dataset):
    """
    Returns the number of target value '1' of the dataset (only if the other class is '0').
    
    Parameters
    ----------
    dataset : list
        A list of datasets.
    
    Returns
    -------
    int
        The total number of target value '1' in the dataset.
    """
    return np.sum([data.y.sum().item() for data in dataset])
