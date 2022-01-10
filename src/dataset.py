# -*- coding: utf-8 -*-
"""
Created on 31/3/2019
@author: RuihongQiu
"""

import pickle
import torch
import collections
from torch_geometric.data import InMemoryDataset, Data


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for sequence, y in zip(data[0], data[1]):
            # sequence = [1, 2, 3, 2, 4]
            count = collections.Counter(sequence)
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}  # ReId nodes for each session graph
            senders = []  # ReIded all nodes in sequence
            x = []  # unique nodes in temporal order. Origin node ID, starts from 1.
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            num_count = [count[i[0]] for i in x]  # occurrence of unique nodes

            if len(senders) != 1:
                del senders[-1]  # the last item is a receiver
                del receivers[0]  # the first item is a sender

            pair = {}  # occurrence of the edge
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender) + '-' + str(receiver) in pair:
                    pair[str(sender) + '-' + str(receiver)] += 1
                    del senders[i]  # remove duplicate edges from senders and receivers
                    del receivers[i]
                else:
                    pair[str(sender) + '-' + str(receiver)] = 1
                    i += 1

            count = collections.Counter(senders)
            out_degree_inv = [1 / count[i] for i in senders]  # unique edges only

            count = collections.Counter(receivers)
            in_degree_inv = [1 / count[i] for i in receivers]  # unique edges only
            
            in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
            out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)

            edge_count = [pair[str(senders[i]) + '-' + str(receivers[i])] for i in range(len(senders))]  # edge count of each unique sender-receiver pair
            edge_count = torch.tensor(edge_count, dtype=torch.float)

            # senders, receivers = senders + receivers, receivers + senders

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            num_count = torch.tensor(num_count, dtype=torch.float)
            sequence = torch.tensor(sequence, dtype=torch.long)
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
            session_graph = Data(x=x, y=y, num_count=num_count,
                                 edge_index=edge_index, edge_count=edge_count,
                                 sequence=sequence, sequence_len=sequence_len,
                                 in_degree_inv=in_degree_inv, out_degree_inv=out_degree_inv)
            data_list.append(session_graph)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
