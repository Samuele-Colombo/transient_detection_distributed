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

import torch
import torch.nn as nn

import numpy as np

from lib.utils.totals import total_positives, total_len

class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.loader = args.loader
        self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')


    def forward(self, out, labels):
        pred = out.argmax(dim=-1)
        totpos = total_positives(self.loader.dataset)
        totlen = total_len(self.loader.dataset)
        true_positives = torch.logical_and(pred == 1, pred == labels).sum().int()/totpos
        true_negatives = torch.logical_and(pred == 0, pred == labels).sum().int()/(totlen-totpos)
        num_positives = labels.sum().item()
        pos_frac = num_positives / len(labels)
        neg_frac = 1. - pos_frac
        assert not np.isnan(pos_frac) and neg_frac >= 0.
        if pos_frac == 0: # in this case placeholder parameters must be enforced to avoid unwanted behavior
            pos_frac = neg_frac = 0.5
            true_positives = 1.
        addloss = (true_positives*true_negatives)**(-0.5) - 1 # scares the model out of giving a constant answer
        loss = self.loss_fn(out, labels, weight=torch.tensor([pos_frac, neg_frac])) + addloss
        assert not torch.isnan(loss.detach()), f"out: {out}\nlabels: {labels}\nLoss: {loss}\nWeight: {pos_frac}"
        return loss


def get_loss(args):
	
    return Loss(args)
