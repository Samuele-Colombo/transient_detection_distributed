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

import torch
from torch_geometric.nn import GCNConv

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation_function=torch.nn.functional.relu):
        """
        Initialize the GCNClassifier model.

        Parameters
        ----------
        num_layers : int
            The number of GCN layers to use in the model.
        input_dim : int
            The input dimension of the data.
        hidden_dim : int
            The hidden dimension of the GCN layers.
        output_dim : int
            The output dimension of the model (i.e. the number of classes).
        activation_function : function, optional
            The activation function of each layer. Default is `torch.nn.functional.relu`.

        """
        super(GCNClassifier, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function

    def forward(self, x, edge_index, dropout_rate=0.5):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data, with shape (batch_size, input_dim).
        edge_index : torch.Tensor
            The edge indices, with shape (2, num_edges).
        dropout_rate : float, optional
            The dropout rate. Default is 0.5.

        Returns
        -------
        torch.Tensor
            The model's output, with shape (batch_size, output_dim).
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation_function(x)
            x = torch.nn.functional.dropout(x, p=dropout_rate, training=self.training)
        x = self.lin(x)
        return x

def get_model(args):

    num_layers = args.num_layers
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    activation_function = args.activation_function

    model = GCNClassifier(num_layers, input_dim, hidden_dim, output_dim, activation_function)

    return model
