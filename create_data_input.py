
import json
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from utils import *
from aminoacids_infor import *
import re
import os
import numpy as np
import pickle
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch
from scipy.spatial import cKDTree
from typing import Dict, Any
import pandas as pd


AA_3TO1 = { "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H",
            "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
            "TYR": "Y", "VAL": "V", "LIG": "X"}#LIG IS THE LIGAND

PERIODIC_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                     'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni',
                     'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb','Unknown']

HYBRIDIZATION = ['S','SP','SP2','SP2D','SP3','SP3D','OTHER','UNSPECIFIED']

BOND_TYPES = ["UNSPECIFIED", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", "AROMATIC", "IONIC", "HYDROGEN",
                  "BondType.THREECENTER", "DATIVEONE", "DATIVE", "DATIVEL", "DATIVER", "OTHER", "PEPTIDE", "hbond",
                  "weak_hbond", "xbond", "ionic", "metal", "aromatic", "hydrophobic", "carbonyl", "polar", "weak_polar",
                  "CARBONPI", "CATIONPI", "METSULPHURPI", "EF", "FT"]

ATOM_TO_IDX = {e:i for i,e in enumerate(PERIODIC_ELEMENTS)}
AA_TO_IDX   = {a:i for i,a in enumerate(AA_3TO1.keys())}
HYB_TO_IDX  = {h:i for i,h in enumerate(HYBRIDIZATION)}
BOND_TO_IDX = {b:i for i,b in enumerate(BOND_TYPES)}


def atom_features(atom, features, resname = "LIG"):
    """Extracts features for an atom"""
    if features == 0:
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),PERIODIC_ELEMENTS) +
                     [atom.GetDegree()/10.0] + #one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                     [atom.GetTotalNumHs()/10.0] + #one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                     [atom.GetIsAromatic()] +
                     one_of_k_encoding_unk(resname, list(AA_3TO1.keys())) +
                  #   one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) +
                  #   [float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0] +
                     one_of_k_encoding_unk(atom.GetHybridization().name, HYBRIDIZATION)
                     )

    elif features == 1:
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),PERIODIC_ELEMENTS) +
                        [atom.GetDegree() / 10.0] +  # one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        [atom.GetTotalNumHs() / 10.0] +  # one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        [atom.GetIsAromatic()] +
                        one_of_k_encoding_unk(resname, list(AA_3TO1.keys())) +
                        [atom.GetImplicitValence() / 10.0]   #one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                        )


def one_of_k_encoding(x, allowable_set):
    """Creates a one-hot encoding for an element in an allowable set."""
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set{allowable_set}")
    return [x == s for s in allowable_set]


def bond_features(bond, protein_interactions=False):
    """Extracts features for an atom"""

    if not protein_interactions:
        bond_types = BOND_TYPES[0:14] #from PEPTIDE are non-covalent
    else:
        bond_types = BOND_TYPES
    return np.array(one_of_k_encoding_unk(bond, bond_types)) #Added peptide and the others in lowercase


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_ligand_from_sdf(path, removeHs= True):
    mol_supplier = Chem.SDMolSupplier(path, removeHs=removeHs)
    mol = mol_supplier[0]
    if mol is None:
        raise ValueError(f"Invalid SDF file or molecule at: {path}")
    return mol


def extract_features_and_indices(mol, node_feature_selection):
    features = []
    index_H_atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            features.append(atom_features(atom, node_feature_selection))
        else:
            index_H_atoms.append(i)
    return features, index_H_atoms


def add_coordinates_from_pdb(ligand_sdf_path, mol, features, ligand_name_pdb, ligand_name_conversion,min_coord, max_coord):
    pdb_path = os.path.join(
        ligand_sdf_path.split('\\Ligand')[0],
        'Protein','Protein_PDB',
        f'{ligand_sdf_path.split("\\")[-1].split("_")[0]}_Protein.pdb'
    )
    pdb_cif_path = os.path.join(
        ligand_sdf_path.split('\\Ligand')[0],
        'Ligand', 'Ligand_CIF',
        f'{ligand_sdf_path.split("\\")[-1].split("_")[0]}_ligand.cif'
    )
    ligand_pdb = ligand_name_conversion.loc[ligand_name_conversion['PDB_ID'] == ligand_name_pdb.split("-")[0], 'Ligand'].tolist()[0]
    ligand_atom_names_pdb_file, ligand_coords = get_atom_names_for_residue(pdb_cif_path, ligand_pdb, min_coord, max_coord)

    if len(ligand_atom_names_pdb_file) == 0 or len(ligand_coords) == 0:
        return 0, 0

    features_with_coord = [np.concatenate([f, c])  for f, c in zip(features, ligand_coords)]
    return features_with_coord, ligand_atom_names_pdb_file



def normalize_3d_coord(coord, min_coord, max_coord):
    """    Normalize a 3D coordinate using min-max normalization.   """

    coord = np.array(coord, dtype=float)
    min_coord = np.array(min_coord, dtype=float)
    max_coord = np.array(max_coord, dtype=float)

    # Avoid division by zero
    range_coord = max_coord - min_coord
    range_coord[range_coord == 0] = 1.0

    return (coord - min_coord) / range_coord


def add_coordinates_from_sdf(conf, features, min_coord, max_coord):
    features_with_coord = []
    for i, f in enumerate(features):
        pos = conf.GetAtomPosition(i)
        coords = normalize_3d_coord([pos.x, pos.y, pos.z], min_coord, max_coord)
        f = np.concatenate([f, coords])
        features_with_coord.append(f)
    return features_with_coord


def build_edges(mol, index_H_atoms, protein_interactions, num_nodes):

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in index_H_atoms and b not in index_H_atoms:
            if a < num_nodes and b < num_nodes:
                edges.append([a, b])
                bond_f = bond_features(bond.GetBondType().name, protein_interactions)
                edge_features.append(bond_f / sum(bond_f))

    return edges, edge_features


def add_distance_to_edges(features_coord, edges, edge_features):
    """  Append Euclidean distances between nodes to edge features."""

    edge_features_updated = []

    for i, (src_idx, tgt_idx) in enumerate(edges):
        src_coords = features_coord[src_idx][-3:]  # x,y,z of source node
        tgt_coords = features_coord[tgt_idx][-3:]  # x,y,z of target node

        # Compute Euclidean distance
        distance = np.linalg.norm(src_coords - tgt_coords)

        # Append distance to the edge feature vector
        updated_features = np.append(edge_features[i], distance)
        edge_features_updated.append(updated_features)

    return edge_features_updated



def smile_to_graph(ligand_sdf_path, protein_interactions, coord, edge_distance,pdb_file, node_feature_selection, ligand_name_conversion,min_coord, max_coord):
    """Converts a SDF string to a graph representation."""
    mol = get_ligand_from_sdf(ligand_sdf_path, removeHs=False)
    conf = mol.GetConformer()

    ligand_name_pdb = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNK"

    rdPartialCharges.ComputeGasteigerCharges(mol)  # Compute Gasteiger charges

    features, index_H_atoms = extract_features_and_indices(mol, node_feature_selection)

    ligand_atom_names_pdb_file = None # will be filled if the coordinates used are exctracted from the PDB file

    if coord or edge_distance:
        if pdb_file:
            features_coord, ligand_atom_names_pdb_file = add_coordinates_from_pdb(ligand_sdf_path, mol, features, ligand_name_pdb, ligand_name_conversion,min_coord, max_coord)
            if features_coord != 0:
                features = features_coord
            elif features_coord == 0:
                return 0, 0, 0
        else:
            features_coord = add_coordinates_from_sdf(conf, features, min_coord, max_coord)
            if features_coord != 0:
                features = features_coord
            elif features_coord == 0:
                return 0, 0, 0

    edges, edge_features = build_edges(mol, index_H_atoms, protein_interactions, len(features))
    if edge_distance:
        edge_features = add_distance_to_edges(features_coord, edges, edge_features)
    num_atoms = len(features) # mol.GetNumAtoms()

    return [num_atoms, features, edges, edge_features], ligand_name_pdb, ligand_atom_names_pdb_file  # edge_index


def clean_atom_name(atom_name):
    # Match the prefix (e.g., C, N), followed by digits, discard trailing letters
    match = re.match(r'^([A-Z]+)(\d+)', atom_name)
    return match.group(1) + match.group(2) if match else atom_name


def get_atom_names_for_residue(pdb_path, ligand_pdb,min_coord, max_coord):
    atom_names = []
    coords = []
    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                #  if line[21] == "A":
                #res_name = line[17:20].strip()
                res_name =  line.split(' ')[6]
                if res_name == ligand_pdb:
                    split_line = line.split()
                   # atom_name = split_line[2]
                    atom_name = line.split(' ')[4]
                    if not atom_name.startswith('H'):
                        atom_name = clean_atom_name(atom_name)
                        if atom_name not in atom_names:
                            atom_names.append(atom_name)
                            xyz = normalize_3d_coord([split_line[-6], split_line[-5], split_line[-4]], min_coord, max_coord)
                            coords.append(xyz)
    return atom_names, coords


def get_binding_residues(pdb_path, ligand_coords, dist=5.0):
    """Get residues within `dist` Å of ligand."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    atoms = [a for a in structure[0].get_atoms() if a.element != "H"]
    ns = NeighborSearch(atoms)
    residues = set()

    for coord in ligand_coords:
        for res in ns.search(coord, dist, level="R"):
            if res.get_resname() in AA_3TO1 and res.get_parent().id == 'A':  # filter out water molecules and chain A only
                residues.add(res)

    return sorted(residues, key=lambda r: r.id[1])


def filter_interactions(entry, dist, chain='A'):
    """    Filters molecular interaction entries based on multiple criteria.    """
    accepted_contacts = {
        "hbond", "weak_hbond", "xbond", "ionic", "metal", "aromatic",
        "hydrophobic", "carbonyl", "polar", "weak_polar"}

    accepted_types = {"atom-plane", "plane-plane", "group-plane"}

    amino_acid_types = list(AA_3TO1.keys())

    entities = entry.get("interacting_entities", "")
    contact_types = set(entry.get("contact", []))
    interaction_type = entry.get("type", "")
    bgn = entry.get("bgn", {})
    row = None
    # Apply initial filters
    if (
            entities == "INTER"
            and bgn.get("auth_asym_id", "") == chain
            and entry["end"].get("auth_asym_id") == chain
    ):
        # Check contact and interaction type
        if contact_types & accepted_contacts or interaction_type in accepted_types:
            # Extract relevant information
            bgn_resname = bgn.get("label_comp_id", "")
            end_resname = entry["end"].get("label_comp_id", "")
           # bgn_seq_id = bgn.get("auth_seq_id", -999)
           # end_seq_id = entry["end"].get("auth_seq_id", -999)

            # Check if residues are amino acids
            bgn_is_residue = bgn_resname in amino_acid_types
            end_is_residue = end_resname in amino_acid_types

            # Skip if neither is an amino acid residue
            if not (bgn_is_residue or end_is_residue):
                return None

            # Distance check
            distance = entry.get("distance")
            if distance is None or distance > dist:
                return None

            # Build row depending on which side is the residue
            if bgn_is_residue:
                row = {
                    'ligand_atom': entry["end"].get("auth_atom_id"),
                    'protein_residue': bgn.get("label_comp_id"),
                    'protein_id': bgn.get("auth_seq_id"),
                    'protein_atom': bgn.get("auth_atom_id"),
                    "contact": ";".join(entry.get("contact", [])),
                    "distance": distance,
                    "type": interaction_type
                    }
            else:  # end_is_residue
                row = {
                    'ligand_atom': bgn.get("auth_atom_id"),
                    'protein_residue': entry["end"].get("label_comp_id"),
                    'protein_id': entry["end"].get("auth_seq_id"),
                    'protein_atom': entry["end"].get("auth_atom_id"),
                    "contact": ";".join(entry.get("contact", [])),
                    "distance": distance,
                    "type": interaction_type
                }

    return row


def ligand_protein_interactions(interaction_path, dist=5.0, chain="A"):#,binding_residues):

    try:
        with open(interaction_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        data = None
    except FileNotFoundError:
        print(f"File not found: {interaction_path}")
        data = None
    except Exception as e:
        print(f"Unexpected error reading JSON: {e}")
        data = None

    filtered_rows = []

   # binding_residues_id = []
   # for b in binding_residues:
#    binding_residues_id.append(b.get_id()[1])

    if data is not None:
        for entry in data:
            row = filter_interactions(entry, dist)
            if row:
                filtered_rows.append(row)
        if len(filtered_rows) == 0:
            # Define the chain priorities
            chain_options = ["AAA_2", "AAA", "B"]

            for chain in chain_options:
                for entry in data:
                    row = filter_interactions(entry, dist, chain=chain) if chain else filter_interactions(entry, dist)
                    if row:
                        filtered_rows.append(row)
                # Stop checking other chains if we already have results
                if filtered_rows:
                    break
    return filtered_rows

def sort_residues_by_number(residue_list):
    return sorted(residue_list, key=lambda x: int(x.split('_')[-1]))


def get_atom_id(atom):
    """Generate a unique ID for an atom."""
    res = atom.get_parent()
    return f"{res.get_resname()}_{res.get_id()[1]}_{res.get_parent().id}_{atom.get_id()}"


def calculate_implicit_valences(atom_id, degree, num_h):
    result = standard_valences[atom_id] - degree - num_h
    return result


def atom_protein_features(atom, node_feature_selection, resname = "LIG"):

    # Use atom.element or atom.get_id()
    element = atom.element if hasattr(atom, 'element') else 'Unknown'
    element_vec = one_of_k_encoding_unk(element, PERIODIC_ELEMENTS)
    # Degree, TotalNumHs, ImplicitValence, Aromatic might be unavailable in BioPython atom
    # You can add placeholders or skip or calculate some if possible
    degree = [aminoacids[resname][atom.id]["degree"] / 10.0]
  #  degree_vec = one_of_k_encoding(degree, [0, 1, 2, 3, 4, 5, 6,7,8,9,10])
    hydrogens = [aminoacids[resname][atom.id]["num_h"]/10.0]
  #  num_h = one_of_k_encoding_unk(hydrogens, [0, 1, 2, 3, 4, 5, 6,7,8,9,10])
    # Aromaticity can be approximated from residue/atom name, but BioPython doesn't provide it directly
    is_aromatic = int('PHE' in ['PHE', 'TYR', 'TRP', 'HIS'])
    resname_vec = one_of_k_encoding_unk(resname, list(AA_3TO1.keys()))

    if node_feature_selection == 0:
       # formal_charge = [0] * 7  # placeholder
       # gastieg_charge = 0.0
        hybrid =  aminoacids[resname][atom.id]["hybrid"]
        hybridization =  one_of_k_encoding_unk(hybrid, ['S', 'SP', 'SP2', 'SP2D','SP3','SP3D', 'OTHER','UNSPECIFIED'])
        #print(resname_vec)
        features = element_vec + degree + hydrogens + [is_aromatic] + resname_vec  + hybridization#element_vec + degree_vec + num_h + [is_aromatic] + resname_vec + formal_charge + [gastieg_charge] + hybridization


    elif node_feature_selection == 1:
        implicit_val = [calculate_implicit_valences(atom.id, degree, hydrogens)/10.0]
      #  implicit_valence_vec = one_of_k_encoding_unk(implicit_val, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Encode amino acid residue
        # Combine all features
        features = element_vec + degree + hydrogens + [is_aromatic] + resname_vec + implicit_val

    # return np.array(features)
    return features



def find_consecutive_residue_pairs(sorted_residues):
    """
    Given a list of sorted residues like ['ARG_7', 'PHE_8', 'ASP_12'],
    return pairs where residue numbers are consecutive.
    """
    pairs = []
    resseqs = [int(r.split('_')[1]) for r in sorted_residues]

    for i in range(len(resseqs) - 1):
        if resseqs[i] + 1 == resseqs[i + 1]:
            pairs.append((sorted_residues[i], sorted_residues[i + 1]))

    return pairs


def extract_atoms_and_bonds(pdb_path, interaction_contacts, ligand_coord, edge_distance, node_feature_selection, protein_interactions, min_coord, max_coord):
    """Extract atoms and intra-/inter-residue bonds."""

    aa_in_interaction_contacts = list({f"{ic['protein_residue']}_{ic['protein_id']}" for ic in interaction_contacts})

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    sorted_residues = sort_residues_by_number(aa_in_interaction_contacts)

    atom_nodes = {}
    atom_index_map = {}
    atoms_list = []
    all_atom_features = []
    coords = []
    edges = []
    edge_features = []
    # Access atoms from the first model
    model = structure[0]

    count = 0
    for chain in model:
        for i, ic in enumerate(sorted_residues):
            resname, resnum = ic.split('_')
            resnum = int(resnum)

            try:
                protein_residue = chain.child_dict[(' ', resnum, ' ')]
            except KeyError:
                continue  # skip missing residues

            atoms = [a for a in protein_residue if a.element != "H"]
            coords_res = []
            atom_number = 0
            for a in atoms:
                atom_nodes[get_atom_id(a)] = a
                atom_index_map[get_atom_id(a)] = count + atom_number
                pos = normalize_3d_coord(a.get_coord(), min_coord, max_coord)
                coords_res.append(pos)
                atoms_list.append(a)
                features = np.array(atom_protein_features(a, node_feature_selection, resname)).astype(int).tolist()
                if ligand_coord:
                    full_features = features + pos.tolist()
                elif edge_distance:
                    if not ligand_coord:
                        full_features = features + pos.tolist()
                else:
                    full_features = features
                all_atom_features.append(np.array(full_features))
                atom_number += 1
            coords_res = np.array(coords_res)
            coords.append(coords_res)

            # Build KDTree for fast neighbor search
            tree = cKDTree(coords_res)

            # Query neighbors within cutoff (e.g., 4.0 Å)
            cutoff = 1.9  # 1.6
            neighbors = tree.query_ball_tree(tree, cutoff)
            # neighbors[i] is a list of indices within cutoff of atoms_list[i]

            for j, nbrs in enumerate(neighbors):
                if len(nbrs) > 1:
                    all_atom_features[count + j][44 + len(nbrs) - 1] = 1
                    atom_j = atoms_list[j]
                    # atom_j_id = get_atom_id(atom_j)
                    close_atoms = [atoms_list[x] for x in nbrs if x != j]
                    for c_a in close_atoms:
                        sorted_bond = sorted((j + count, atoms_list.index(c_a) + count))
                        if sorted_bond not in edges:
                            edges.append(sorted_bond)
                            try:
                                new_bond = aminoacid_bonds[resname][atom_j.id][c_a.id]
                            except KeyError:
                                # If the key doesn't exist, just skip or assign None or handle as needed
                          #      print(resname, atom_j.id, c_a.id)
                                new_bond = "SINGLE"

                            if isinstance(new_bond, str):
                                e_features = bond_features(new_bond, protein_interactions)
                            else:
                                e_features = bond_features("SINGLE", protein_interactions)
                            if edge_distance and  not ligand_coord:
                                coord1 = normalize_3d_coord(atom_j.get_coord(), min_coord, max_coord)
                                coord2 = normalize_3d_coord(c_a.get_coord(), min_coord, max_coord)
                                distance = float(np.linalg.norm(coord1 - coord2))
                                e_features = np.concatenate([e_features, [distance]])

                            edge_features.append(e_features)

                else:
                    all_atom_features[count + j][44] = 1
                    #    print(f'error in {res} {res.get_id()[1]} atom name {close_atoms}')
            if i > 0:
                prev_resname, prev_resnum = sorted_residues[i - 1].split('_')
                prev_resnum = int(prev_resnum)
                if resnum == prev_resnum + 1:
                    try:
                        prev_residue = chain.child_dict[(' ', prev_resnum, ' ')]
                        atom_C = prev_residue["C"]
                        atom_N = protein_residue["N"]
                    except KeyError:
                        atom_C, atom_N = None, None

                    if atom_C is not None and atom_N is not None:
                        dist = np.linalg.norm(atom_C.get_coord() - atom_N.get_coord())
                        dist_normalized = np.linalg.norm(normalize_3d_coord(atom_C.get_coord(), min_coord, max_coord) - normalize_3d_coord(atom_N.get_coord(), min_coord, max_coord))
                        if 1.2 < dist < 1.45:
                            id_C = get_atom_id(atom_C)
                            id_N = get_atom_id(atom_N)

                            idx_C = atom_index_map.get(id_C)
                            idx_N = atom_index_map.get(id_N)

                            if idx_C is not None and idx_N is not None:
                                edges.append([idx_C, idx_N])
                                e_features = bond_features("PEPTIDE", True)
                                if edge_distance and not ligand_coord:
                                    e_features = np.concatenate([e_features, [dist_normalized]])
                                edge_features.append(e_features)

            count += len(atoms)

    return atom_nodes, all_atom_features, atom_index_map, edges, edge_features


def non_covalent_interactions(interaction_contacts, graph, ligand_atom_names_pdb_file,
                                   all_atom_features, atom_index_map, edges, edge_features,
                                   length_ligand_coords, ligand_coord, edge_distance):
    if not graph[1]:
        print('Empty graph ')
        return False
    if len(all_atom_features) == 0:
        print('No aminoacid residues interacting with the ligand')
        return False
    if len(graph[1][0]) != len(all_atom_features[0]):
        if edge_features:
            if len(graph[1][0]) != len(all_atom_features[0]) -3:
                print('ligand and protein node features have different dimensions')
                return graph
        else:
            print('ligand and protein node features have different dimensions')
            return graph
    if len(length_ligand_coords) != len(ligand_atom_names_pdb_file):
        print('ligand from the SDF and PDB files have different dimensions')
        return graph

    ligand_count = len(ligand_atom_names_pdb_file)
    new_edges = []
    new_edge_features = []

    # Append protein atoms and update total node count

    graph[1].extend(all_atom_features)
    graph[0] = len(graph[1])

    if len(interaction_contacts) != 0:
        for interaction in interaction_contacts:

            for atom_x in interaction['ligand_atom'].split(','):
              #  if len(interaction['ligand_atom'].split(',')) > 1:
              #      print(interaction)
            #    atom_x = interaction['ligand_atom'].split(',')[0]
                if atom_x not in ligand_atom_names_pdb_file:
                    print(f'{atom_x} not in listed ligand atom names')
                    continue
                pos_ligand = ligand_atom_names_pdb_file.index(atom_x)
                for atom_y in interaction['protein_atom'].split(','):
               #     if len(interaction['protein_atom'].split(',')) > 1:
               #         rer = 0

                    try:
                        atom_key = f"{interaction['protein_residue']}_{interaction['protein_id']}_A_{atom_y}"
                        raw_pos_protein = atom_index_map[atom_key]
                    except KeyError:
                        continue
                    pos_protein = raw_pos_protein + ligand_count
                    for contact in interaction['contact'].split(';'):
                        edge_contact = [pos_ligand, pos_protein]
                        e_features = bond_features(contact, True)
                        if edge_distance and not ligand_coord:
                            coord_ligand = length_ligand_coords[pos_ligand]
                            coord_protein = graph[1][pos_protein][-3:]
                            dist = np.linalg.norm(coord_ligand - coord_protein)
                            e_features = np.concatenate([e_features, [dist]])
                        edge_contact_features = np.array(e_features)
                        if edge_contact not in new_edges or (edge_distance and not ligand_coord):
                            new_edges.append(edge_contact)
                            new_edge_features.append(edge_contact_features)
                        else:
                            index = new_edges.index(edge_contact)
                            if not np.all(new_edge_features[index] == edge_contact_features): #new_edge_features[index] != edge_contact_features:
                                updated_edge_features = np.array([e or c for e, c in
                                                         zip(new_edge_features[index], edge_contact_features)])
                                new_edge_features[index] = updated_edge_features

    if edge_distance and not ligand_coord:
        if len(graph[1][0]) == len(all_atom_features[0]) - 3:
            graph[1][len(length_ligand_coords):] = list(map(lambda a: a[:-3], graph[1][len(length_ligand_coords):]))
    # Append intra-protein edges with index offset
    for e in edges:
        graph[2].append([e[0] + ligand_count, e[1] + ligand_count])
    graph[2].extend(new_edges)
    graph[3].extend(edge_features)
   # new_edge_features = [np.array(f) for f in new_edge_features]
    graph[3].extend(new_edge_features)

    # Sanity check for index range
    all_edge_indices = [i for edge in graph[2] for i in edge]
    if max(all_edge_indices) > len(graph[1]):
        print(f"[ERROR] Invalid edge index! Max index: {max(all_edge_indices)}, total nodes: {len(graph[1])}")

    return graph


def save_input_data(data, path):
    """Saves a dictionary to a .pkl file."""
    full_path = os.path.join(path, "data.pkl")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        pickle.dump(data, f)
    print(f'[INFO] Input data saved at: {full_path}')


def open_file_input_data(path):
    """Loads a dictionary from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def input_model_data(common_keys,ligands_dict: Dict[str, str], ligand_coord: bool, protein_interaction: bool, edge_distance:bool,
                     pdb_files: Dict[str, str],node_feature_selection: str, DATA_PATH: str, DATA_OUT: str,
                     save_input_data_format: bool, min_coord: list, max_coord:list) -> Dict[str, Any]:
    """ Processes ligand and protein data to generate graph representations.
       Returns: smile_graph: Dictionary of processed ligand graph data.  """

    smile_graph = {}
    count_ligands = 0
    path_conversion = os.path.join(DATA_PATH, "380_Oct_2025_TOT.csv")
    ligand_name_conversion = pd.read_csv(path_conversion, sep=',', header=0, engine='python')
    for ligand_pdb_name in common_keys:
        count_ligands +=1
        ligand_sdf_path = os.path.join(DATA_PATH, 'Ligand', 'Ligand_SDF', f'{ligand_pdb_name}_ligand.sdf')

        if not os.path.exists(ligand_sdf_path):
            print(f'[WARN] Ligand SDF file not found: {ligand_sdf_path}')
            continue

        graph, ligand_name_pdb, ligand_atom_names_pdb_file = smile_to_graph(ligand_sdf_path, protein_interaction,
                                                                            ligand_coord, edge_distance, pdb_files, node_feature_selection, ligand_name_conversion, min_coord, max_coord)
        if graph == 0:
            print(f'[SKIP] {ligand_pdb_name}: No ligand coordinates found in PDB')
            continue

        max_edge_idx = max(max(edge) for edge in graph[2])
        if max_edge_idx >= graph[0]:
            print(f'[ERROR] {ligand_pdb_name}: Edge index out of range ({max_edge_idx} >= {graph[0]})')
            continue

        if protein_interaction:

            pdb_path = os.path.join(DATA_PATH, 'Protein', 'Protein_PDB', f'{ligand_pdb_name}_protein.pdb')

            interaction_path = os.path.join(DATA_PATH, 'Binding','Interaction_JSON', f'{ligand_pdb_name.lower()}_ligand.json')


            if ligand_atom_names_pdb_file is None:
                ligand_atom_names_pdb_file, ligand_coords_pdb = get_atom_names_for_residue(pdb_path, ligand_name_conversion, min_coord, max_coord)
            else:
                ligand_coords_pdb = [arr[-3:] for arr in graph[1]]

            if edge_distance and not ligand_coord: #remove 3d coord ligand node features
                graph[1] = [a[:-3] for a in graph[1]]

            print(f'[INFO] Processing {ligand_pdb_name} ({count_ligands})')

            interaction_contacts = ligand_protein_interactions(interaction_path, dist=7.0)
            if not interaction_contacts:
                print(f'[INFO] No protein interactions found for {ligand_name_pdb}')
                continue

            atom_nodes, all_atom_features, atom_index_map, edges, edge_features = extract_atoms_and_bonds(pdb_path, interaction_contacts, ligand_coord, edge_distance,node_feature_selection, protein_interaction, min_coord, max_coord)
            graph = non_covalent_interactions(interaction_contacts, graph, ligand_atom_names_pdb_file,
                                               all_atom_features, atom_index_map, edges, edge_features,
                                                ligand_coords_pdb, ligand_coord, edge_distance)
        smile_graph[ligands_dict[ligand_pdb_name]] = graph


    if save_input_data_format:
        save_input_data(smile_graph, DATA_OUT)

    return smile_graph


def filter_missing_smiles(drugs, labels, prots, smile_graph):
    """Filter out samples whose SMILES are not in the smile_graph."""
  #  indices_to_remove = sorted([i for i, drug in enumerate(train_drugs) if drug not in smile_graph], reverse=True)
    #for id in indeces:
    #    train_drugs = np.delete(train_drugs, id, axis=0)
    #    train_Y = np.delete(train_Y, id, axis=0)
    if drugs is not None:
        if drugs.size > 0:
            indices_to_remove = [i for i, drug in enumerate(drugs) if drug not in list(smile_graph.keys())] # [i for i, drug in enumerate(drugs) if drug not in smile_graph]
            print(f"Filtered indices (missing graphs): {indices_to_remove}")
            drugs = np.delete(drugs, indices_to_remove, axis=0)
            labels = np.delete(labels, indices_to_remove, axis=0)
            if prots is not None:
                prots = np.delete(prots, indices_to_remove, axis=0)
    else:
        drugs, labels, prots = None, None, None
    return drugs, labels, prots


def create_dataset(path, drugs, prots, labels, smile_graph, root='results_main'):
    """Wraps TestbedDataset creation."""
    if drugs is not None:
        if drugs.size > 0:
            return TestbedDataset(root=root, dataset=path, xd=drugs, xt=prots, y=labels, smile_graph=smile_graph)


def get_smiles_affinity_data(ligands_dict, indices, affinity_df):
    if indices:
        ligands_list = list(ligands_dict.values())
        drugs = []
        Y = []
        for i in indices:
            ligand_name = affinity_df.iloc[i, 0]  # column 0 = ligand name
            affinity = affinity_df.iloc[i, 1]     # column 1 = affinity
            if ligand_name in ligands_dict:
                drugs.append(ligands_dict[ligand_name])
                Y.append(affinity)
            else:
                # Ligand missing in dictionary
                print(f"Warning: ligand {ligand_name} not found in ligands_dict, skipping.")
        drugs = np.array(drugs)
        Y = np.array(Y)
    else:
        drugs,Y  = None, None
    return drugs, Y


def sequence_vector(normalize=True):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}  # values from 1 to 24
    max_seq_len = 306 #1000

    if normalize:
        min_val = 1
        max_val = len(seq_voc)
        seq_dict = {k: (v - min_val) / (max_val - min_val) for k, v in seq_dict.items()}

    return seq_dict, max_seq_len


def seq_cat(prot, seq_dict, max_seq_len):
    """Converts protein sequence into categorical format."""
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
        #x[i] = seq_dict.get(ch, 0)
    return x
   # return np.zeros(max_seq_len)
   # random = [7, 8, 9, 3, 5, 8, 8, 4, 9, 4, 2, 7, 2, 8, 8, 1, 9, 0, 6, 8, 9, 1, 8, 1, 5, 2, 4, 0, 4, 2, 8, 4, 8, 7, 9, 9, 6, 0, 0, 0, 7, 0, 5, 2, 8, 7, 5, 2, 1, 7, 2, 3, 9, 3, 1, 2, 4, 3, 6, 6, 6, 0, 7, 4, 7, 4, 6, 1, 5, 3, 2, 4, 6, 4, 5, 5, 2, 0, 2, 1, 7, 3, 4, 7, 3, 3, 9, 1, 2, 5, 3, 7, 8, 2, 0, 3, 6, 5, 5, 5, 8, 1, 4, 9, 0, 6, 5, 7, 9, 9, 0, 5, 3, 1, 0, 4, 9, 5, 6, 0, 3, 7, 8, 5, 5, 7, 4, 7, 5, 4, 2, 9, 8, 2, 9, 9, 6, 2, 4, 6, 1, 7, 3, 2, 6, 2, 1, 1, 5, 3, 9, 9, 6, 6, 2, 9, 9, 6, 8, 8, 6, 5, 7, 5, 6, 3, 0, 5, 4, 5, 6, 9, 7, 6, 2, 9, 8, 6, 7, 6, 2, 4, 1, 3, 5, 5, 3, 3, 4, 0, 4, 6, 4, 0, 8, 7, 8, 8, 0, 4, 9, 6, 0, 5, 0, 7, 8, 1, 9, 4, 2, 4, 2, 3, 0, 7, 9, 6, 1, 7, 3, 7, 0, 3, 3, 6, 6, 7, 8, 4, 8, 3, 8, 6, 1, 1, 8, 1, 3, 7, 8, 2, 3, 9, 0, 9, 4, 4, 0, 1, 2, 1, 6, 0, 3, 3, 1, 1, 4, 7, 2, 9, 5, 0, 4, 2, 3, 9, 1, 0, 1, 0, 8, 3, 1, 8, 6, 2, 0, 2, 1, 0, 4, 7, 5, 2, 9, 1, 7, 1, 9, 5, 3, 8, 5, 2, 0, 8, 1, 1, 3, 0, 5, 4, 5, 7]
   # return np.array(random)



def prepare_datasets(data_f, affinity,DATA_OUT, model_name, smile_graph, ligands_dict, protein_as_sequence, protein_sequences_dict, train_valid_test_name= 'train'):
    """    Prepares PyTorch Geometric datasets from ligand SMILES and affinity CSVs.    """

    #base_name = ligands_file.split(".txt")[0]
    #suffix = ligands_name.split("_", 1)[1]
 #   dataset_name = DATA_OUT.split("\\")[-1]

    file_data = os.path.join(DATA_OUT, f'{model_name}_{train_valid_test_name}.pt')

    if not (os.path.isfile(file_data)):
        # Extract SMILES and affinity values

        data_drugs, data_Y =  get_smiles_affinity_data(ligands_dict, data_f, affinity)
        data_prots  = None  # Placeholder for potential protein inputs

        if protein_as_sequence:
            if len(protein_sequences_dict) > 0:
                seq_dict, max_seq_len = sequence_vector()
                if len(protein_sequences_dict) == 1:
                    only_sequence = list(protein_sequences_dict.values())[0]
                    if data_f:
                        data_prots = np.asarray([seq_cat(only_sequence, seq_dict, max_seq_len) for _ in data_f])
                else:
                    prot_keys = list(protein_sequences_dict.keys())
                    if data_f:
                        data_prots = np.asarray([seq_cat(protein_sequences_dict[prot_keys[i]],seq_dict, max_seq_len) for i in data_f])#np.asarray([seq_cat(protein_sequences_dict.iloc[i, 1],seq_dict, max_seq_len) for i in train_f])

        #MOVE THE FILTERING PREVIOUSLY BECAUSE IN KFOLD FIRST THE INDEX 363 IS MANUALLY REMOVED TO AVOID ERRORS IN GET_SMILES_AFFINITY_DATA
        # Filter missing ligands if needed
        data_drugs, data_Y, data_prots = filter_missing_smiles(data_drugs, data_Y, data_prots, smile_graph)


        print(f'Preparing dataset in PyTorch format!')
        dataset_data = create_dataset(file_data, data_drugs, data_prots, data_Y, smile_graph)

        print(f'Created: {file_data}')

    else:
        print(f'Loading existing datasets: {file_data}')
        dataset_data = TestbedDataset(root='results_main', dataset=file_data)

    return dataset_data