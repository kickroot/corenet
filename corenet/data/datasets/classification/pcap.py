from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets import (BaseDataset)
import numpy as np
import os
import torch
from scapy.all import rdpcap
from collections import Counter
from enum import Enum

class PCAPMode(Enum):
    PER_STREAM = "PER_STREAM"
    PER_PACKET = "PER_PACKET"

@DATASET_REGISTRY.register(name="pcap", type="classification")
class PCAPDataset(BaseDataset):

    def _load_pcap_per_stream(self, pcap_file_path):
        """Load a PCAP file and convert its raw bytes into a tensor.  This builds a tensor PER STREAM"""
        try:
            packets = rdpcap(pcap_file_path)
            np.random.shuffle(list(packets))
            raw_bytes = b"".join(bytes(pkt) for pkt in packets)
            return [torch.tensor(list(raw_bytes), dtype=torch.uint8)]
        except Exception as e:
            print(f"Error loading {pcap_file_path}: {e}")
            return None
        
    def _load_pcap_per_packet(self, pcap_file_path):
        """Load a PCAP file and convert its raw bytes into a tensor. This builds a tensor PER PACKET"""
        try:            
            packets = rdpcap(pcap_file_path)
            return [torch.tensor(list(bytes(pkt)), dtype=torch.uint8) for pkt in packets]
        except Exception as e:
            print(f"Error loading {pcap_file_path}: {e}")
            return None        

    def _load_pcap_dataset(self, pcap_root_folder):
        """
        Recursively loads PCAP files into a dictionary categorized by their subfolder name.  This builds a tensor PER STREAM

        Args:
            pcap_root_folder (str): The root folder containing category subfolders.

        Returns:
            list: A list of dict elements consisting of `label` and `data` keys, with label mapping to a string and data to a tensor.
        """
        collate_function = self._load_pcap_per_stream if PCAPMode(self.collate_mode) == PCAPMode.PER_STREAM else self._load_pcap_per_packet
        print(f"****** Loading PCAP Dataset from {pcap_root_folder} using function {collate_function.__name__}*********")
        label_map = {"benign": 0, "malicious": 1}

        items = []
        # Walk through each subfolder
        for category in os.listdir(pcap_root_folder):
            category_path = os.path.join(pcap_root_folder, category)

            if os.path.isdir(category_path):  # Ensure it's a folder    

                # Iterate over files in the subfolder
                for filename in os.listdir(category_path):
                    if filename.endswith(".pcap"):  # Ensure only PCAP files are processed
                        file_path = os.path.join(category_path, filename)

                        for tensor in collate_function(file_path):
                            items.append({'label': label_map[category], 'data': tensor})
    
        # Let's count & log the class counts
        label_counts = Counter(d["label"] for d in items)
        print(f" Label counts: {label_counts}")
        return items


    def __init__(self, opts, is_training: bool = True, is_evaluation: bool = False):
        super().__init__(opts)
        dataset_root = getattr(opts, "dataset.root_train")
        print(f"collate_mode={getattr(opts, 'dataset.pcap_collate_mode')}")
        self.collate_mode = getattr(opts, "dataset.pcap_collate_mode", PCAPMode.PER_STREAM)
        if not is_training and not is_evaluation:
            dataset_root = getattr(opts, "dataset.root_val")

        print(f"DATASET LOADING: is_training={is_training}, is_eval={is_evaluation}, root={dataset_root}")
        self.items = self._load_pcap_dataset(dataset_root)  # Load PCAP file paths



    def __getitem__(self, sample_size_and_index):
        _, _ , idx = sample_size_and_index

        return self.items[idx]

    def __len__(self):
        return len(self.items)