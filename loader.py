from anndata._core.anndata import AnnData
from torch.utils.data import Dataset
import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import warnings


class Matrix(Dataset):
    def __init__(self,
                 adata: AnnData = None,
                 global_graph: Data = None,
                 obs_label_colname: str = "x",
                 augmentation: bool = False,
                 args_augmentation: dict = {}
                 ):

        super().__init__()

        self.adata = adata
        if isinstance(self.adata.X, np.ndarray):
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()

        if self.adata.obs.get(obs_label_colname) is not None:
            self.label = self.adata.obs[obs_label_colname]
            self.unique_label = list(set(self.label))
            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        else:
            self.label = None

        self.augmentation = augmentation
        self.num_cells, self.num_genes = self.adata.shape
        self.args_augmentation = args_augmentation
        self.data_for_augmentation = deepcopy(self.data)
        self.global_graph = global_graph

    def RandomAugmentation(self, sample, index):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for neighbor in NeighborLoader(self.global_graph, num_neighbors=[self.data.shape[0]],
                                        input_nodes=torch.Tensor([index]).to(torch.long)):
                neighbor_idx = neighbor.n_id.numpy()  

        tr = transformation(self.data_for_augmentation, sample, neighbor_idx)

        tr.random_gaussian_noise(self.args_augmentation['noise_percentage'], self.args_augmentation['sigma'],
                                self.args_augmentation['noise_prob'])

        tr.random_gene_dropout(self.args_augmentation['dropout_percentage'], self.args_augmentation['dropout_prob'])

        tr.one_neighbor_crossover(self.args_augmentation['exchange_percentage'],
                                self.args_augmentation['exchange_prob'])

        tr.cell_state_interpolation(self.args_augmentation['interpolation_alpha'],
                                    self.args_augmentation['interpolation_prob'])

        tr.ToTensor()
        return tr.cell_profile

    def __getitem__(self, index):
        sample = self.data[index]

        if self.label is not None:
            label = self.label_encoder[self.label.iloc[index]]
        else:
            label = -1
        if self.augmentation:
            sample_1 = self.RandomAugmentation(sample, index)
            sample = [sample, sample_1]
        return sample, index, label

    def __len__(self):
        return self.adata.X.shape[0]


class transformation():
    def __init__(self,
                 dataset,  
                 cell_profile,  
                 neighbor_idx):  

        self.dataset = dataset
        self.cell_profile = deepcopy(cell_profile)
        self.gene_num = len(self.cell_profile)
        self.cell_num = len(self.dataset)
        self.neighbor_idx = neighbor_idx

    def build_mask(self, masked_percentage: float):

        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool),
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        np.random.shuffle(mask)
        return mask


    def random_gaussian_noise(self,
                              noise_percentage: float = 0.7,
                              sigma: float = 0.2,
                              apply_noise_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_noise_prob:
            mask = self.build_mask(noise_percentage)
            noise = np.random.normal(0, sigma, int(self.gene_num * noise_percentage))
            self.cell_profile[mask] += noise


    def random_gene_dropout(self,
                            dropout_percentage: float = 0.2,
                            apply_dropout_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_dropout_prob:
            mask = self.build_mask(dropout_percentage)
            self.cell_profile[mask] = 0


    def one_neighbor_crossover(self,
                               cross_percentage: float = 0.3,
                               apply_cross_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_cross_prob:
            list_idx = np.random.randint(1, len(self.neighbor_idx))
            cross_idx = self.neighbor_idx[list_idx]
            cross_instance = self.dataset[cross_idx]
            mask = self.build_mask(cross_percentage)
            tmp = cross_instance[mask].copy()
            cross_instance[mask], self.cell_profile[mask] = self.cell_profile[mask], tmp


    def cell_state_interpolation(self,
                                 alpha: float = 0.5,
                                 apply_interpolation_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_interpolation_prob:
            list_idx = np.random.randint(1, len(self.neighbor_idx))
            neighbor_cell = self.dataset[self.neighbor_idx[list_idx]]
            self.cell_profile = alpha * self.cell_profile + (1 - alpha) * neighbor_cell

    def ToTensor(self):
        self.cell_profile = torch.from_numpy(self.cell_profile)