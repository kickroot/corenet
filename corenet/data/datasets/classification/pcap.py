from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets import (BaseDataset)
import numpy as np
import os
import torch
from scapy.all import rdpcap

@DATASET_REGISTRY.register(name="pcap", type="classification")
class PCAPDataset(BaseDataset):

    def _load_pcap_as_tensor(self, pcap_path):
        """Load a PCAP file and convert its raw bytes into a tensor."""
        try:
            packets = rdpcap(pcap_path)
            np.random.shuffle(list(packets))
            raw_bytes = b"".join(bytes(pkt) for pkt in packets)
            byte_tensor = torch.tensor(list(raw_bytes), dtype=torch.uint8)
            return byte_tensor
        except Exception as e:
            print(f"Error loading {pcap_path}: {e}")
            return None
        
    def _load_pcaps_as_tensor(self, pcap_path):
        """Load a PCAP file and convert its raw bytes into a tensor."""
        try:            
            packets = rdpcap(pcap_path)
            return [torch.tensor(list(bytes(pkt)), dtype=torch.uint8) for pkt in packets]
            # np.random.shuffle(list(packets))
            # raw_bytes = b"".join(bytes(pkt) for pkt in packets)
            # byte_tensor = torch.tensor(list(raw_bytes), dtype=torch.uint8)
            # return byte_tensor
        except Exception as e:
            print(f"Error loading {pcap_path}: {e}")
            return None        

    def _load_pcap_dataset(self, pcap_root_folder):
        """
        Recursively loads PCAP files into a dictionary categorized by their subfolder name.

        Args:
            pcap_root_folder (str): The root folder containing category subfolders.

        Returns:
            list: A list of dict elements consisting of `label` and `data` keys, with label mapping to a string and data to a tensor.
        """
        print(f"****** Loading PCAP Dataset from {pcap_root_folder} *********")
        label_map = {"benign": 0, "malicious": 1}

        items = []
        # Walk through each subfolder
        for category in os.listdir(pcap_root_folder):
            category_path = os.path.join(pcap_root_folder, category)

            if os.path.isdir(category_path):  # Ensure it's a folder    
                # pcap_dict[category] = []  # Initialize list for this category

                # Iterate over files in the subfolder
                for filename in os.listdir(category_path):
                    if filename.endswith(".pcap"):  # Ensure only PCAP files are processed
                        file_path = os.path.join(category_path, filename)
                        for tensor in self._load_pcaps_as_tensor(file_path):
                            items.append({'label': label_map[category], 'data': tensor})
                        # if tensor is not None:
                        #     # print(f"Adding {file_path} to {category}")
                        #     items.append({'label': label_map[category], 'data': tensor})
                        #     # pcap_dict[category].append(tensor)
        return items


    def __init__(self, opts, is_training: bool = True, is_evaluation: bool = False):
        super().__init__(opts)
        # print(opts)
        # print(f"Dataset initialized: {opts.dataset.split}, Path: {opts.dataset.root}")
        # print(f"PCAPDataset: opts={opts}, is_training={is_training}, is_evaluation={is_evaluation}")
        dataset_root = getattr(opts, "dataset.root_train")
        if not is_training and not is_evaluation:
            dataset_root = getattr(opts, "dataset.root_val")

        print(f"DATASET LOADING: is_training={is_training}, is_eval={is_evaluation}, root={dataset_root}")
        self.items = self._load_pcap_dataset(dataset_root)  # Load PCAP file paths



    def __getitem__(self, sample_size_and_index):
        _, _ , idx = sample_size_and_index

        return self.items[idx]

    def __len__(self):
        return len(self.items)