# convert itp file to lmp file
import numpy as np
import readlammpsdata as rld

def read_info(itp_file_atomsinfo):
  info = np.loadtxt(itp_file_atomsinfo,skiprows=18,dtype="str")
  atomtype = info[:,0].tolist()
  atommass = info[:,1].tolist()
  charge = info[:,2].tolist() # e
  real_sigma = info[:,7].tolist() # (Ang.) 
  real_epsilon = info[:,8].tolist() # (kcal/mol)
  mass_dict = dict(zip(atomtype, atommass))
  idtype = [i+1 for i in range(len(atomtype))]
  type_dict = dict(zip(atomtype, idtype))
  real_sigma_dict = dict(zip(atomtype, real_sigma))
  real_epsilon_dict = dict(zip(atomtype, real_epsilon))
  return mass_dict,type_dict,real_sigma_dict,real_epsilon_dict

def generate_Masses(mass_dict):
  masses = list(mass_dict.values())
  atom_typeid = [i+1 for i in range(len(masses))]
  atom_typeele = list(mass_dict.keys())
  Masses = np.array([atom_typeid,masses,atom_typeele]).T
  Jings = np.array(['#' for _ in range(len(atom_typeele))])
  Masses = np.insert(Masses, 2, Jings, axis=1)
  return Masses

def generate_PairCoeffs(real_sigma_dict,real_epsilon_dict):
  real_sigma = list(real_sigma_dict.values())
  real_epsilon = list(real_epsilon_dict.values())
  atom_typeid = [i+1 for i in range(len(real_sigma))]
  atom_typeele = list(real_sigma_dict.keys())
  PairCoeffs = np.array([atom_typeid,real_epsilon,real_sigma,atom_typeele]).T
  Jings = np.array(['#' for _ in range(len(atom_typeele))])
  PairCoeffs = np.insert(PairCoeffs, 3, Jings, axis=1)
  return PairCoeffs

def generate_BondCoefs(itp_file,atoms_info):
  atomid = atoms_info[:,0]
  atom_typeeleid = atoms_info[:,4]
  atomtype_dict = dict(zip(atomid, atom_typeeleid))
  bonds_info = np.loadtxt(itp_file,skiprows=itp_file_para[1][0],max_rows=itp_file_para[1][1],dtype="str")
  real_ro = bonds_info[:,6].tolist()
  real_kr = bonds_info[:,7].tolist()
  bond_id = [i+1 for i in range(len(real_kr))]
  Jings = np.array(['#' for _ in range(len(real_kr))])
  
  ai = bonds_info[:,0].tolist()
  aj = bonds_info[:,1].tolist()
  bondtypes = bonds_info[:,8].tolist()
  unique_bondtypes = list(dict.fromkeys(bondtypes))
  # print(unique_bondtypes)
  bondtype = [str(i+1) for i in range(len(unique_bondtypes))]
  bondtype_dict = dict(zip(unique_bondtypes, bondtype))
  bonds_id = [i+1 for i in range(len(ai))]
  bondtypeid = []
  for i in range(len(bondtypes)):
    bondtypeid.append(bondtype_dict[bondtypes[i]])
  Bonds = np.array([bonds_id,bondtypeid,ai,aj]).T

  bonds_connect = []
  for i in range(len(ai)):
    bonds_connect.append(f"{atomtype_dict[ai[i]]}-{atomtype_dict[aj[i]]}")
  BondCoeffs = np.array([bond_id,real_kr,real_ro,Jings,bonds_connect]).T
  return BondCoeffs, Bonds


def generate_AngleCoefs(itp_file,atoms_info):
  atomid = atoms_info[:,0]
  atom_typeeleid = atoms_info[:,4]
  atomtype_dict = dict(zip(atomid, atom_typeeleid))
  angles_info = np.loadtxt(itp_file,skiprows=itp_file_para[2][0],max_rows=itp_file_para[2][1],dtype="str")
  # print(angles_info)
  real_c0 = angles_info[:,7].tolist()
  real_c1 = angles_info[:,8].tolist()
  angle_id = [i+1 for i in range(len(real_c1))]
  Jings = np.array(['#' for _ in range(len(real_c1))])
  
  ai = angles_info[:,0].tolist()
  aj = angles_info[:,1].tolist()
  ak = angles_info[:,2].tolist()

  angletypes = angles_info[:,9].tolist()
  unique_angletypes = list(dict.fromkeys(angletypes))
  # print(unique_angletypes)
  angletype = [str(i+1) for i in range(len(unique_angletypes))]
  angletype_dict = dict(zip(unique_angletypes, angletype))
  angles_id = [i+1 for i in range(len(ai))]
  angletypeid = []
  for i in range(len(angletypes)):
    angletypeid.append(angletype_dict[angletypes[i]])
  Angles = np.array([angles_id,angletypeid,ai,aj,ak]).T

  angles_connect = []
  for i in range(len(ai)):
    angles_connect.append(f"{atomtype_dict[ai[i]]}-{atomtype_dict[aj[i]]}-{atomtype_dict[ak[i]]}")
  AngleCoeffs = np.array([angle_id,real_c1,real_c0,Jings,angles_connect]).T
  return AngleCoeffs,Angles


def generate_DihedralCoefs(itp_file,atoms_info):
  atomid = atoms_info[:,0]
  atom_typeeleid = atoms_info[:,4]
  atomtype_dict = dict(zip(atomid, atom_typeeleid))
  dihedrals_info = np.loadtxt(itp_file,skiprows=itp_file_para[3][0],max_rows=itp_file_para[3][1],usecols=([i for i in range(15)]),dtype="str")
  # print(dihedrals_info)
  real_V1 = dihedrals_info[:,10].tolist()
  real_V2 = dihedrals_info[:,11].tolist()
  real_V3 = dihedrals_info[:,12].tolist()
  real_V4 = dihedrals_info[:,13].tolist()
  dihedral_id = [i+1 for i in range(len(real_V1))]
  Jings = np.array(['#' for _ in range(len(real_V1))])
  
  ai = dihedrals_info[:,0].tolist()
  aj = dihedrals_info[:,1].tolist()
  ak = dihedrals_info[:,2].tolist()
  al = dihedrals_info[:,3].tolist()

  dihedraltypes = dihedrals_info[:,14].tolist()
  unique_dihedraltypes = list(dict.fromkeys(dihedraltypes))
  dihedraltype = [str(i+1) for i in range(len(unique_dihedraltypes))]
  dihedraltype_dict = dict(zip(unique_dihedraltypes, dihedraltype))
  dihedrals_id = [i+1 for i in range(len(ai))]
  dihedraltypeid = []
  for i in range(len(dihedraltypes)):
    dihedraltypeid.append(dihedraltype_dict[dihedraltypes[i]])
  Dihedrals = np.array([dihedrals_id,dihedraltypeid,ai,aj,ak,al]).T

  dihedrals_connect = []
  for i in range(len(ai)):
    dihedrals_connect.append(f"{atomtype_dict[ai[i]]}-{atomtype_dict[aj[i]]}-{atomtype_dict[ak[i]]}-{atomtype_dict[al[i]]}")
  DihedralCoeffs = np.array([dihedral_id,real_V1,real_V2,real_V3,real_V4,Jings,dihedrals_connect]).T
  
  return DihedralCoeffs, Dihedrals


def generate_Atoms(pdb_file,atoms_info,type_dict):
  atoms_coord = np.loadtxt(pdb_file,skiprows=pdb_file_para[0],max_rows=pdb_file_para[1],dtype="str")
  atom_typeid = [i+1 for i in range(len(atoms_coord))]
  mole_typeid = atoms_info[:,2].tolist()
  
  atomeles = atoms_info[:,1].tolist()
  atomtypes = []
  # print(type_dict)
  for ele in  atomeles:
    atomtypes.append(type_dict[ele])
  charges = atoms_info[:,6].tolist()
  atomsx = atoms_coord[:,5].tolist()
  atomsy = atoms_coord[:,6].tolist()
  atomsz = atoms_coord[:,7].tolist()
  atom_typeeleid = atoms_coord[:,2].tolist()
  
  Atoms = np.array([atom_typeid,mole_typeid,atomtypes, charges,atomsx,atomsy,atomsz,atom_typeeleid]).T
  Jings = np.array(['#' for _ in range(len(atomsz))])
  Atoms = np.insert(Atoms, 7, Jings, axis=1)
  return  Atoms

def write_lmp(lmp,Masses,
    PairCoeffs,BondCoeffs,AngleCoeffs,DihedralCoefs,
    Atoms,Bonds,Angles,Dihedrals):
  natoms = len(Atoms)
  nbonds = len(Bonds)
  nangles = len(Angles)
  ndihedrals = len(Dihedrals)

  natomtypes = len(Masses)
  nbondtypes = len(BondCoeffs)
  nangletypes = len(AngleCoeffs)
  ndihedraltypes = len(DihedralCoefs)

  Masses_str = rld.array2str(Masses)
  PairCoeffs_str = rld.array2str(PairCoeffs)
  BondCoeffs_str = rld.array2str(BondCoeffs)
  AngleCoeffs_str = rld.array2str(AngleCoeffs)
  DihedralCoefs_str = rld.array2str(DihedralCoefs)
  Atoms_str = rld.array2str(Atoms)
  Bonds_str = rld.array2str(Bonds)
  Angles_str = rld.array2str(Angles)
  Dihedrals_str = rld.array2str(Dihedrals)

  with open(lmp,"w") as f:
    f.write("LAMMPS data file from itp file - (Written by Dongsheng Chen)\n")
    f.write(f"{natoms} atoms\n")
    f.write(f"{nbonds} bonds\n")
    f.write(f"{nangles} angles\n")
    f.write(f"{ndihedrals} dihedrals\n\n")

    f.write(f"{natomtypes} atom types\n")
    f.write(f"{nbondtypes} bond types\n")
    f.write(f"{nangletypes} angle types\n")
    f.write(f"{ndihedraltypes} dihedral types\n\n")

    f.write(f"-10   10 xlo xhi\n")
    f.write(f"-10   10 ylo yhi\n")
    f.write(f"-10   10 zlo zhi\n\n")

    f.write(f"Masses")
    f.write(Masses_str)
    f.write(f"Pair Coeffs")
    f.write(PairCoeffs_str)
    f.write(f"Bond Coeffs")
    f.write(BondCoeffs_str)
    f.write(f"Angle Coeffs")
    f.write(AngleCoeffs_str)
    f.write(f"Dihedral Coeffs")
    f.write(DihedralCoefs_str)
    f.write(f"Atoms")
    f.write(Atoms_str)
    f.write(f"Bonds")
    f.write(Bonds_str)
    f.write(f"Angles")
    f.write(Angles_str)
    f.write(f"Dihedrals")
    f.write(Dihedrals_str)

  return


if __name__ == '__main__':
  # 需要两个itp文件，一个pdb文件："EMIM.itp"、"EMIM_atomtypes.itp"
  # 还需要手动给出参数：itp_file_para是"EMIM.itp"文件中atoms、bonds、angles等的起始行数
  # itp_file_atomsinfo_para是"EMIM_atomtypes.itp"文件中atomtypes的起始行数
  # pdb_file_para是"EMIM.pdb"文件中坐标的起始行数
  itp_file = "EMIM.itp" # "DCA.itp" # "EMIM.itp"
  itp_file_para =  [[22,19],[44,19],[66,33],[102,46]] # [[22,19],[44,19],[66,33],[102,46]] # [[22,5],[30,4],[37,3],[43,2]]
  itp_file_atomsinfo = "EMIM_atomtypes.itp" # "DCA_atomtypes.itp" # "EMIM_atomtypes.itp"
  itp_file_atomsinfo_para = [18,11] # [18,11] # [18,3]
  pdb_file = "EMIM.pdb" # "DCA.pdb" # "EMIM.pdb"
  pdb_file_para = [21,19] #  [21,19] # [21,5]

  lmp = "EMIM.data" # "DCA.data" # "EMIM.data"




  mass_dict,type_dict,real_sigma_dict,real_epsilon_dict = read_info(itp_file_atomsinfo)
  Masses = generate_Masses(mass_dict)
  PairCoeffs = generate_PairCoeffs(real_sigma_dict,real_epsilon_dict)
  atoms_info = np.loadtxt(itp_file,skiprows=itp_file_para[0][0],max_rows=itp_file_para[0][1],dtype="str")
  BondCoeffs, Bonds = generate_BondCoefs(itp_file,atoms_info)
  AngleCoeffs, Angles = generate_AngleCoefs(itp_file,atoms_info)
  DihedralCoefs, Dihedrals = generate_DihedralCoefs(itp_file,atoms_info)
  Atoms = generate_Atoms(pdb_file,atoms_info,type_dict)
  write_lmp(lmp,Masses,
    PairCoeffs,BondCoeffs,AngleCoeffs,DihedralCoefs,
    Atoms,Bonds,Angles,Dihedrals)



  


