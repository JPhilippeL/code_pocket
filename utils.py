import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA_Data
from torch import load, LongTensor, Tensor, FloatTensor, from_numpy
from torch import save as torch_save

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = load(self.processed_paths[0],  weights_only=False) #si peta aqui, posar ', weights_only=False'

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        if self.dataset.endswith('.pt'):
            return [self.dataset]
        else:
            return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data


    def process(self, xd, xt, y,smile_graph):

        # ---------- Sanity checks ----------
        if xt is not None:
            assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        else:
            assert (len(xd) == len(y)), "The two lists must be the same length!"

        data_list = []
        data_len = len(xd)

        y_list = y.iloc[:, 1].tolist()
        for i in range(data_len):
          #  print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            if xt is not None:
                target = xt[i]
            label_value = float(y_list[i])

            try:
                # convert SMILES to molecular representation using rdkit
                #c_size, features, edge_index, edge_features = smile_graph[smiles]

                graph = smile_graph[smiles]

                c_size = graph['num_atoms']
                features = graph['features'] # torch.Tensor
                edge_index = graph['edges']
                edge_features = graph['edge_features']

                # make the graph ready for PyTorch Geometrics GCN algorithms:
                #features = np.array(features)

                try:
                   # label_value = float(labels)
                    try:
                       edge_attr = edge_features.float()
                    except AttributeError:
                       edge_attr = FloatTensor(np.array(edge_features, dtype=np.float32))

                    if isinstance(features, Tensor):
                        x_data = FloatTensor(np.array( features.detach().cpu().numpy(), dtype=np.float32))
                    else:
                        x_data = FloatTensor(np.array(features))

                    if len(edge_index) == 0:
                        print("Skipping empty pocket graph")
                        continue

                    GCNData = DATA_Data(
                        x = x_data,
                        edge_index=LongTensor(edge_index).transpose(1, 0),
                        y=Tensor([label_value]),
                        edge_attr= edge_attr)  #FloatTensor(np.array(edge_features, dtype=np.float32))) #x=Tensor(features), edge_attr= Tensor(edge_features),  edge_attr= FloatTensor(np.array(edge_features, dtype=np.float32)).transpose(1, 0)
                    if xt is not None:
                        if isinstance(target, np.ndarray):
                            GCNData.target = FloatTensor([target]) #from_numpy(target).unsqueeze(0)
                        else:
                            GCNData.target = LongTensor(np.array(target, dtype=np.int64)).unsqueeze(0) #LongTensor(np.array(target)).unsqueeze(0)
                        #GCNData.target = LongTensor([target])
                    GCNData.__setitem__('c_size', LongTensor([c_size]))
                    # append graph, label and target sequence to data list
                    data_list.append(GCNData)

                except ValueError as e:
                    print(f"[Skipping] Invalid label '{y.iloc[0, i]}': {e}")

            except KeyError as e:
                print(f'SMILE {e} not used due to KeyError')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:

        os.makedirs(os.path.join(self.processed_paths[0].split('\\')[0] + '\\',
                         os.path.join(*self.processed_paths[0].split('\\')[1:-1])),exist_ok=True)
        torch_save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
