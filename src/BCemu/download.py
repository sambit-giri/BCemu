import os, time
import wget
# import pkg_resources
from importlib.resources import files

def download_emulators():
    print('Downloading emulators...')

    # Define the package name where the input_data folder is located
    package_name = "BCemu"

    # Path to the emulator folder
    path_to_emul_folder = str(files(package_name) / 'input_data')

    # Ensure the input_data folder exists
    os.makedirs(path_to_emul_folder, exist_ok=True)

    # Paths to local emulator files
    path_to_emu0_file   = os.path.join(path_to_emul_folder, 'kpls_emulator_z0_nComp3.pkl')
    path_to_emu0p5_file = os.path.join(path_to_emul_folder, 'kpls_emulator_z0p5_nComp3.pkl')
    path_to_emu1_file   = os.path.join(path_to_emul_folder, 'kpls_emulator_z1_nComp3.pkl')
    path_to_emu1p5_file = os.path.join(path_to_emul_folder, 'kpls_emulator_z1p5_nComp3.pkl')
    path_to_emu2_file   = os.path.join(path_to_emul_folder, 'kpls_emulator_z2_nComp3.pkl')

    # URLs for the emulator files
    link_to_emu0_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z0_nComp3.pkl'
    link_to_emu0p5_file = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z0p5_nComp3.pkl'
    link_to_emu1_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z1_nComp3.pkl'
    link_to_emu1p5_file = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z1p5_nComp3.pkl'
    link_to_emu2_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z2_nComp3.pkl'

    # Download the files
    print('at z = 0.0');   wget.download(link_to_emu0_file,   path_to_emu0_file)
    print('\nat z = 0.5'); wget.download(link_to_emu0p5_file, path_to_emu0p5_file)
    print('\nat z = 1.0'); wget.download(link_to_emu1_file,   path_to_emu1_file)
    print('\nat z = 1.5'); wget.download(link_to_emu1p5_file, path_to_emu1p5_file)
    print('\nat z = 2.0'); wget.download(link_to_emu2_file,   path_to_emu2_file)

    print('...done')
    return None