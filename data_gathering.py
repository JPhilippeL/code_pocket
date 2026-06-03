
import os
from rdkit import Chem
import shutil
from Bio.PDB import PDBList, PDBParser, PPBuilder, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from split_data import read_fold_from_file

three_to_one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def get_name_from_file(file_name):
    return os.path.splitext(file_name)[0].split('_')[0]
    #return file_name


def get_smile_from_SDF(sdf_file_path):
    """Extracts the canonical SMILES from an SDF file"""
    try:
        suppl = Chem.SDMolSupplier(sdf_file_path)
        for mol in suppl:
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Chem.SanitizeMolError:
        print(f"Error sanitizing molecule in file: {sdf_file_path}")
    return None


def save_ligands_to_file(sequences, file_name, results_path):
    """Saves ligand sequences to a specified text file."""

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    #os.makedirs(results_path.split(f'\\{file_name}')[0], exist_ok=True)
    with open(results_path, 'w') as f:
        f.write("{")
        if sequences:
            first_name, first_sequence = next(iter(sequences.items()))
            f.write(f'"{first_name}": "{first_sequence}"')
            for name, sequence in list(sequences.items())[1:]:
                f.write(f', "{name}": "{sequence}"')
        f.write("}")
    print(f"Sequences written to {results_path}")


def process_ligands(ligands_file, DATA_PATH, DATA_OUT):
    """Processes ligands from SDF files and saves them to a text file"""
    try:
        results_path = os.path.join(DATA_OUT, ligands_file)
        sequences = {}
        error_ligands = []
        if not os.path.exists(results_path):
            ligands_folder = os.path.join(DATA_PATH, "Ligand", "Ligand_SDF")
            for file_name in os.listdir(ligands_folder):
                if file_name.endswith('.sdf'):
                    name = get_name_from_file(file_name)
                    sequence = get_smile_from_SDF(os.path.join(ligands_folder, file_name))
                    if sequence:
                        sequences[name] = sequence
                    else:
                        error_ligands.append(name)
            save_ligands_to_file(sequences, ligands_file, results_path)
            if len(error_ligands) != 0:
                error_path = os.path.join(DATA_OUT, "ligand_error.txt")
                save_ligands_to_file({ligand: '' for ligand in error_ligands}, "ligand_error.txt", error_path)
        else:
            print(f'File at {results_path} already exists, check if you wish to make any change on it')
            sequences = read_fold_from_file(results_path)
            error_path = os.path.join(DATA_OUT, "ligand_error.txt")
            if os.path.exists(error_path):
                error_ligands = read_fold_from_file(error_path)
                if len(error_ligands) > 0:
                    error_ligands = list(error_ligands.keys())
        return sequences, error_ligands
    except Exception as e:
        print(f"Error processing ligands: {str(e)}")


def get_seqres_sequence_chain_A(pdb_file_path):
    seqres_seq = []
    seen_seqres = False
    with open(pdb_file_path, 'r') as handle:
        for line in handle:
            if line.startswith("SEQRES"):
                seen_seqres = True
                chain_id = line[11]
                if chain_id == 'A':
                    residues = line[19:].split()
                    for res in residues:
                        try:
                            aa = three_to_one[res]
                        except KeyError:
                            aa = 'X'  # Use 'X' for unknown or non-standard residues
                        seqres_seq.append(aa)
            elif seen_seqres:
                break
    return ''.join(seqres_seq)


def get_sequence_from_ent(ent_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", ent_file)

    ppb = PPBuilder()
    sequences = []

    for peptide in ppb.build_peptides(structure):
        sequences.append(str(peptide.get_sequence()))

    return sequences


def get_sequence_from_PDB(PDB_id,folder_path, DATA_OUT):
    """Extracts sequence from a local PDB file"""
    pdbl = PDBList()
    try:
        # Suppress warnings from Biopython about structure issues
        warnings.simplefilter('ignore', PDBConstructionWarning)

        proteins_path = os.path.join(DATA_OUT,"proteins_pdb_python")
        pdbl.retrieve_pdb_file(PDB_id, pdir=proteins_path, file_format='pdb')
     #   parser = PDBParser(QUIET=True)
        file = f'pdb{PDB_id.lower()}.ent'
        abs_path = f'{folder_path}\\{PDB_id}_protein.pdb'
        sequence = get_sequence_from_ent(os.path.join(proteins_path,file))
       # sequence_chain_A = get_seqres_sequence_chain_A(abs_path)
       # print(f"SEQRES sequence for chain A of {PDB_id}:\n{sequence_chain_A}")
        return  sequence
    except Exception as e:
        print(f"Error processing local PDB file {PDB_id}: {str(e)}")
        return None


def convert_3_to_1(aa_list):
    sequence = ""
    for res in aa_list:
        try:
            sequence += three_to_one(res.upper())
        except KeyError:
            sequence += "X"  # Unknown/Non-standard residue
    return sequence


def parse_cif_file_dict(filepath):
    mmcif_dict = MMCIF2Dict(filepath)
    seq = mmcif_dict.get('_struct_ref.pdbx_seq_one_letter_code')[0]
    if not seq:
        seq_list = mmcif_dict.get('_entity_poly_seq.mon_id')
        seq = convert_3_to_1(seq_list)
    else:
        seq = ''.join(seq.split('\n'))
    return seq


def parse_cif_file(filepath):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(filepath), filepath)
        sequence = ''
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    res_id = residue.id
                    for atom in residue:
                        m = 'ooo'
        return sequence
    except Exception as e:
        print(f" Failed to parse {filepath}: {e}")


def process_protein_sequences(DATA_PATH, DATA_OUT):

    sequences = {}
    folder_path = os.path.join(DATA_PATH, "Protein", "Protein_PDB")

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdb'):
            name = get_name_from_file(file_name)
            print(name)
            sequence = get_sequence_from_PDB(name, folder_path, DATA_OUT)
            if len(sequence) == 0:
                cif_file_path = os.path.join(DATA_PATH, "Protein", "Protein_CIF", f'{name}_protein.cif')
                sequence = parse_cif_file_dict(cif_file_path)
            if sequence:
                sequences[name] = sequence
            else:
                  #  print(sequences)
                error_folder = os.path.join(DATA_PATH, "error_proteins")
                os.makedirs(error_folder, exist_ok=True)
                error_file_path = os.path.join(folder_path, file_name)
                shutil.move(error_file_path, os.path.join(error_folder, file_name))
                print(f"Moved {file_name} to {error_folder} due to processing error.")

    return sequences


def processed_protein_directory(results_path, DATA_PATH, DATA_OUT,proteins_file):
    print(f'File at {results_path} already exists, check if you wish to make any change on it')
    sequences = read_fold_from_file(results_path)
    if len(sequences) == 0:
        sequences = process_protein_sequences(DATA_PATH, DATA_OUT)
        #  os.remove(results_path)
        save_ligands_to_file(sequences, proteins_file, os.path.join(DATA_OUT, proteins_file))
    return sequences


def process_proteins(proteins_file, DATA_PATH, DATA_OUT, protein_as_sequence):
    """Process proteins from PDB files and saves the to a text file"""
    try:
        results_path = os.path.join(DATA_OUT, proteins_file)
        if  os.path.exists(results_path): # Case 1: results already exist → load/process from file
            sequences = processed_protein_directory(results_path, DATA_PATH, DATA_OUT, proteins_file)
        else:   # Case 2: results do NOT exist → process and save
            sequences = process_protein_sequences(DATA_PATH, DATA_OUT)
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            save_ligands_to_file(sequences, proteins_file,os.path.join(DATA_OUT, proteins_file))
        return sequences
    except Exception as e:
        print(f"Error processing proteins: {str(e)}")
        return None
