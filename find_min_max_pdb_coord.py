import os
from Bio.PDB import PDBParser
import numpy as np


def get_global_min_max_coords(pdb_dir):
    parser = PDBParser(QUIET=True)

    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])

    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_dir, filename)
            try:
                structure = parser.get_structure(filename, pdb_path)
                for atom in structure.get_atoms():
                    coord = atom.get_coord()
                    min_coords = np.minimum(min_coords, coord)
                    max_coords = np.maximum(max_coords, coord)
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")

    return min_coords, max_coords

if __name__ == "__main__":
    pdb_directory = r"C:\Users\natal\code_binding_data_and_results\data\URV_Database_2025_Octubre\URV_Database_2025_Octubre\Protein\Protein_PDB"
    min_xyz, max_xyz = get_global_min_max_coords(pdb_directory)

    print("Minimum coordinates (x, y, z):", min_xyz)
    print("Maximum coordinates (x, y, z):", max_xyz)