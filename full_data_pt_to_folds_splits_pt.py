import torch
import os
from split_data import *
from protein_class_embedding import load_graph

exported_pocket_path = r'C:\Users\natal\code_binding_data_and_results\results\Mpro-URV\2_GINE\E5\pocket_BD\pocket_BD.pkl'
collection_ligands = load_graph(exported_pocket_path)
list_ligands = list(collection_ligands.keys())
folder = r'C:\Users\natal\code_binding_data_and_results\results\Mpro-URV\2_GINE\E5\pocket_BD'
f_name = 'pocket_BD'

data_list = torch.load(os.path.join(folder, f"{f_name}.pt"),weights_only=False)
results_folder = r'C:\Users\natal\code_binding_data_and_results\results\Mpro-URV'
num_folds = 5
split_data_folder = "split_data"
train_folds, valid_folds, test_folds = train_valid_test_folds(list_ligands, num_folds,  0.19,
                                                                        split_data_folder,'train_index_folder.txt',
                                                                       'valid_index_folder.txt','test_index_folder.txt', results_folder,
                                                                        os.path.join(results_folder, 'nothing'))

for fold in range(num_folds):
    train_f = train_folds[fold]
    valid_f = valid_folds[fold]
    test_f = test_folds[fold]

    train_list = [data for data in data_list if data.name in train_f]
    valid_list = [data for data in data_list if data.name in valid_f]
    test_list = [data for data in data_list if data.name in test_f]

    torch.save(train_list, os.path.join(folder, f"{f_name}_train_{fold}.pt"))
    torch.save(valid_list, os.path.join(folder, f"{f_name}_valid_{fold}.pt"))
    torch.save(test_list, os.path.join(folder, f"{f_name}_test_{fold}.pt"))




