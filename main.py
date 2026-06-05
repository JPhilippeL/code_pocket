from ligand_class_embedding import *
from split_data import *
#from fold_training_functions import *
from protein_class_embedding import *

def folder_name(ligand_graph, protein_interaction, protein_as_sequence, include_coords, include_edge_distances):
    word = ''

    if ligand_graph:
        if protein_interaction:
            word += 'pocket'
            if protein_as_sequence:
                word += '_prot_seq'
        else:
            word += 'ligand'
            if protein_as_sequence:
                word += '_prot_seq'
    else:
        if protein_as_sequence:
            word += 'prot_seq'

    if include_coords:
        word += '_3D'
    if include_edge_distances:
        word += '_BD'

    return word


def create_ligand_collection(sdf_directory: str, collection_ligands,  affinity_data, min_coord, max_coord, use_embedding_nodes: bool = True,
                             use_embedding_edges: bool = False, node_feature_selection: int = 0, emb_dim_atom: int = 16,
                             emb_dim_aa: int = 7, emb_dim_hyb: int = 4, emb_dim_bond: int = 3, include_coords: bool = True,
                             include_edge_distances: bool = True, simplified_edge_distances: bool = False):
    """
    Build and return a fully initialized LigandCollection.
    """

    # Load ligands
    collection_ligands.load_from_directory(sdf_directory=sdf_directory, node_feature_selection=node_feature_selection,
                                           use_embedding_nodes=use_embedding_nodes, use_embedding_edges=use_embedding_edges, simplified_edge_distances=simplified_edge_distances)

    # Initialize embedders
    if use_embedding_nodes:
        collection_ligands.initialize_node_embedders(emb_dim_atom=emb_dim_atom,emb_dim_aa=emb_dim_aa, emb_dim_hyb=emb_dim_hyb)

    if use_embedding_edges:
        if not simplified_edge_distances:
            collection_ligands.initialize_edge_embedders(emb_dim_bond=emb_dim_bond)

    # Build graphs (and apply embeddings internally)
    success_count, data_list = collection_ligands.build_all_graphs(include_coords=include_coords,min_coord=min_coord,max_coord=max_coord, affinity_data=affinity_data,
                                        include_edge_distances=include_edge_distances, simplified_edge_distances=simplified_edge_distances)

    return collection_ligands, data_list


def create_pocket_collection(collection_ligands, pockets_interaction, affinity_data,results_folder: str, use_embedding_nodes: bool = False,
                                    use_embedding_edges: bool = False, emb_dim_atom: int = 14,emb_dim_aa: int = 7,
                                    emb_dim_hyb: int = 3, emb_dim_bond: int = 7, include_coords: bool = True, min_coord: Optional[List[float]] = None,
                                    max_coord: Optional[List[float]] = None, include_edge_distances: bool = False,
                                    interaction_json_dir: Optional[str] = None, distance_threshold: float = 7.0,
                                    pocket_by_distance: bool = False, simplified_edge_distances: bool = False):

    # Initialize embedders
    if use_embedding_nodes:
        pockets_interaction.initialize_node_embedders(emb_dim_atom=emb_dim_atom, emb_dim_aa=emb_dim_aa,
                                                     emb_dim_hyb=emb_dim_hyb)

    if use_embedding_edges:
        if not simplified_edge_distances:
            pockets_interaction.initialize_edge_embedders(emb_dim_bond=emb_dim_bond)

    # Build graphs
    ligand_protein_collection, data_list = pockets_interaction.build_all_pockets(collection_ligands=collection_ligands,
                                                                      ligand_coords=include_coords,min_coord=min_coord,
                                                                      max_coord=max_coord, affinity_data=affinity_data,include_edge_distances=include_edge_distances,
                                                                      interaction_json_dir=interaction_json_dir, distance_threshold=distance_threshold,
                                                                      pocket_by_distance=pocket_by_distance, simplified_edge_distances=simplified_edge_distances)

    # Export
    #exported_pocket_path = os.path.join(results_folder, "protein_ligand_graphs.pkl")
    #export_dict_graph_data(ligand_protein_collection, exported_pocket_path)

    return ligand_protein_collection, data_list

def run_pipeline(name,ligand_graph,protein_interaction,protein_as_sequence,ligand_sdf_directory,
                 interaction_json_dir,results_folder,split_data_folder,DATA_PATH,min_coord,max_coord,use_embedding_nodes,
                 use_embedding_edges,include_coords,include_edge_distances,pocket_by_distance,simplified_edge_distances, num_layers,repetitions =1):
    collection_ligands = None
    affinity_data = check_affinity_file(DATA_PATH, True, sep="  ", file_name="pIC50.txt")
    if ligand_graph:
        f_name = folder_name(ligand_graph, False, protein_as_sequence, include_coords, include_edge_distances)
        folder = results_folder.replace(name,f_name)
        collection_ligands = LigandCollection()
        exported_ligand_path_pkl = os.path.join(folder, f"{f_name}.pkl")
        if not os.path.exists(exported_ligand_path_pkl):
            os.makedirs(folder, exist_ok=True)
            collection_ligands, data_list = create_ligand_collection(sdf_directory=ligand_sdf_directory, collection_ligands=collection_ligands, affinity_data=affinity_data,min_coord=min_coord,max_coord=max_coord,
                                                        use_embedding_nodes=use_embedding_nodes,use_embedding_edges=use_embedding_edges,
                                                        include_coords=include_coords, include_edge_distances=include_edge_distances, simplified_edge_distances=simplified_edge_distances)
            # Export to pickle (compatible with existing code)

            collection_ligands.export_graph_data(exported_ligand_path_pkl)
            torch.save(data_list, os.path.join(folder, f"{f_name}.pt"))
        collection_ligands = collection_ligands.load_graph_data(exported_ligand_path_pkl)
        data_list = torch.load(os.path.join(folder, f"{f_name}.pt"),weights_only=False)
            # common_keys = list(set(ligands_dict.keys()) & set(protein_sequences_dict.keys()))

    distance_threshold = 7.0
    #pocket_data_list = torch.load(os.path.join(results_folder, f"{name}.pt"),weights_only=False)

    if protein_interaction:
        exported_pocket_path = os.path.join(results_folder, f"{name}.pkl")
        # Initialize pocket collection
        pockets_interaction = PocketCollection(use_embedding_nodes=use_embedding_nodes,use_embedding_edges=use_embedding_edges, simplified_edge_distances=simplified_edge_distances)
        if not os.path.exists(exported_pocket_path):
            collection_ligands, data_list_pocket = create_pocket_collection(collection_ligands,  pockets_interaction=pockets_interaction,  affinity_data=affinity_data,results_folder= results_folder, use_embedding_nodes= use_embedding_nodes,
                                    use_embedding_edges = use_embedding_edges,include_coords = include_coords, min_coord=min_coord,
                                    max_coord = max_coord, include_edge_distances= include_edge_distances,
                                    interaction_json_dir = interaction_json_dir, distance_threshold= distance_threshold,
                                    pocket_by_distance = pocket_by_distance, simplified_edge_distances=simplified_edge_distances)
            export_dict_graph_data(collection_ligands, exported_pocket_path)
            torch.save(data_list_pocket, os.path.join(results_folder, f"{name}.pt"))
        collection_ligands = load_graph(exported_pocket_path)

   # num_folds = 5


  #  if folder_name(ligand_graph, protein_interaction, protein_as_sequence, include_coords, include_edge_distances) == 'prot_seq':
  #      list_ligands = affinity_data.iloc[:, 0].tolist()
  #  else:
  #      list_ligands = list(collection_ligands.keys())

 #   train_folds, valid_folds, test_folds = train_valid_test_folds(list_ligands, num_folds,  0.19,
 #                                                                       split_data_folder,'train_index_folder.txt',
 #                                                                      'valid_index_folder.txt','test_index_folder.txt', DATA_PATH,
 #                                                                       results_folder)

 #   for fold in range(num_folds):
 #       train_f = train_folds[fold]
 #       valid_f = valid_folds[fold]
 #       test_f = test_folds[fold]

 #       data_out_fold = os.path.join(results_folder, results_folder,f'fold_{fold}')
 #       fold_training(train_f, valid_f, test_f, affinity_data, data_out_fold, name, collection_ligands,
  #                    protein_as_sequence,protein_sequences_dict, protein_interaction, num_layers)


if __name__ == "__main__":

    DATA_PATH = "/home/philippe/Documents/Databases/URV_Database_2025_Octubre"  # Path for MPro data
    min_coord = [-28, -36, -34]
    max_coord = [39, 37, 42]
    use_embedding_nodes = False
    use_embedding_edges = True

    simplified_edge_distances = False

   # include_coords = False
   # include_edge_distances = False
   # protein_as_sequence = False
   # protein_interaction = False
    interaction_json_dir = "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/Binding/Interaction_JSON"
    pocket_by_distance = False
  #  ligand_graph = True

    r_init_folder = "/home/philippe/Documents/Databases"
    for num_layers in [2]:#[2,3,4,5]
        for node_emb, edge_emb, simpl, letter in [[True, False, True, "E3_node_id"]]:#, [True, False, False, "B"], [False, False, False, "D"],[False, True, False, "C"]]: #    for node_emb, edge_emb, simpl, letter in [ [True, False, True, "E"], [False, False, True, "F"],[True, True, False, "A"], [True, False, False, "B"], [False, True, False, "C"],[False, False, False, "D"]]:

            r_folder = os.path.join(r_init_folder, f'{num_layers}_GINE', letter)
            # r_folder = r_init_folder
            ligand_sdf_directory = os.path.join(DATA_PATH, "Ligand", "Ligand_SDF")
            split_data_folder = "split_data"

            repetitions = 5

            for ligand_graph, protein_interaction, protein_as_sequence, include_coords, include_edge_distances in [[True,True,False,False,True], [True,False,False,False,True]]:#,[True,False,False,False,False],[True,True,False,False,False]   [[True,False,False,False,False],[True,False,False,False,True],[True,True,False,False,False],[True,True,False,False,True]]:#[True,False,False,False,False],[True,False,False,False,True],[True,False,False,True,False],[True,False,False,True,True],[True,True,False,False,False],[True,True,False,False,True],[True,True,False,True,False],[True,True,False,True,True]t[True,True,False,False,False],[True,True,False,False,False],[True,True,False,True,False],,[False,False,True,False,False]]:
                name = folder_name(ligand_graph, protein_interaction, protein_as_sequence, include_coords,include_edge_distances)
                results_folder = os.path.join(r_folder, name)

                run_pipeline(name, ligand_graph, protein_interaction, protein_as_sequence,ligand_sdf_directory,
                             interaction_json_dir, results_folder, split_data_folder, DATA_PATH, min_coord, max_coord,node_emb,
                             edge_emb, include_coords, include_edge_distances, pocket_by_distance, simpl,num_layers,repetitions)
            #summary_results_main(r_folder)
