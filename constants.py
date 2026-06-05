from rdkit import Chem
# --- Definiciones de Patrones SMARTS para H-Bonds ---
# Donador: Generalmente N u O con al menos un H unido
HBD_PATTERN = Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]')

# Aceptor: N u O con pares libres disponibles (definición estándar de Lipinski)
HBA_PATTERN = Chem.MolFromSmarts(
    '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]')

# Patrón para enlace rotable: Enlace simple (-), no en anillo (!@), entre átomos no terminales (!D1)
FLEXIBILITY_BOND_PATTERN = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

PERIODIC_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                        'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sb', 'Sn', 'Ag',
                        'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',  'Ni',
                        'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pb', 'Unknown']

HYBRIDIZATION = ['S', 'SP', 'SP2', 'SP2D', 'SP3', 'SP3D', 'OTHER', 'UNSPECIFIED']

BOND_TYPES_COVALENT = ["NONE","SINGLE","DOUBLE","TRIPLE","QUADRUPLE","QUINTUPLE","HEXTUPLE","ONEANDAHALF",
                        "TWOANDAHALF","THREEANDAHALF", "FOURANDAHALF","FIVEANDAHALF","AROMATIC","IONIC", "HYDROGEN",
                        "THREECENTER","DATIVEONE","DATIVE","DATIVEL","DATIVER","OTHER", "ZERO","PEPTIDE"]

BOND_TYPES_NON_COVALENT = ["AMIDERING", "hydrophobic", "CARBONPI","DONORPI","weak_polar","weak_hbond","aromatic",
                            "METSULPHURPI","EF", "vdw_clash", "polar","FE","vdw","hbond","FF","FT","OT","OE","OF","carbonyl","ET",
                            "xbond","HALOGENPI","ionic","CATIONPI"]

BOND_TYPES = {"SINGLE":0,"DOUBLE":1,"TRIPLE":2,"AROMATIC":3, "OTHER": 4, "UNSPECIFIED": 4, "PEPTIDE": 5}

AA_3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
            "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
            "TYR": "Y", "VAL": "V", "LIG": "X"}

ATOM_TO_IDX = {e: i for i, e in enumerate(PERIODIC_ELEMENTS)}
AA_TO_IDX = {a: i for i, a in enumerate(AA_3TO1.keys())}
HYB_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATION)}
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES_COVALENT)}