from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets import BaseDataset
import numpy as np
import os
import torch
from scapy.all import rdpcap
from collections import Counter
from enum import Enum
from pathlib import Path

@DATASET_REGISTRY.register(name="pcap_tuple", type="classification")
class PCAPTupleDataset(BaseDataset):
    def __init__(self, opts, is_training=True, is_evaluation=False):
        super().__init__(opts, is_training=is_training, is_evaluation=is_evaluation)

        self.samples = []
        root_path = Path(self.root)

        for label_dir in sorted(root_path.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                raise ValueError(f"Expected directory names to be integers (class indices), got: {label_dir.name}")

            for file_path in label_dir.glob("*"):
                if file_path.is_file():
                    self.samples.append((file_path, label))

        # self.transforms = self.get_augmentation_transforms()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_size_and_index):
        # if isinstance(sample_size_and_index, (tuple, list)) and len(sample_size_and_index) >= 2:
        #     idx = sample_size_and_index[1]
        # else:
        #     idx = sample_size_and_index
        _, _ , idx = sample_size_and_index
        file_path, label = self.samples[idx]
        data_tensor = self.load_pcap(file_path)
        # print(f"[{self.mode}] idx={idx}, label={label}, file={file_path.name}")
        return {"data": data_tensor, "label": label}

    def load_pcap(self, path: Path):
        packets = rdpcap(str(path))
        raw_bytes = b"".join(bytes(pkt) for pkt in packets)
        return torch.tensor(list(raw_bytes), dtype=torch.int64)


    def get_item_metadata(self, item_idx: int):
        path, label = self.samples[item_idx]
        return {"path": str(path), "label": label}

    def share_dataset_arguments(self):
        # Required for CoreNet to know the number of classes
        num_classes = len(set(label for _, label in self.samples))
        return {"model.classification.n_classes": num_classes}
