import os, time
import wget, pkg_resources

def download_emulators():
	print('Downloading emulators...')

	path_to_emul_folder = pkg_resources.resource_filename('BCemu', 'input_data/')

	path_to_emu0_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z0_nComp3.pkl')
	path_to_emu0p5_file = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z0p5_nComp3.pkl')
	path_to_emu1_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z1_nComp3.pkl')
	path_to_emu1p5_file = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z1p5_nComp3.pkl')
	path_to_emu2_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z2_nComp3.pkl')

	link_to_emu0_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z0_nComp3.pkl'
	link_to_emu0p5_file = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z0p5_nComp3.pkl'
	link_to_emu1_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z1_nComp3.pkl'
	link_to_emu1p5_file = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z1p5_nComp3.pkl'
	link_to_emu2_file   = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/kpls_emulator_z2_nComp3.pkl'

	print('at z = 0.0');   wget.download(link_to_emu0_file,   path_to_emul_folder)
	print('\nat z = 0.5'); wget.download(link_to_emu0p5_file, path_to_emul_folder)
	print('\nat z = 1.0'); wget.download(link_to_emu1_file,   path_to_emul_folder)
	print('\nat z = 1.5'); wget.download(link_to_emu1p5_file, path_to_emul_folder)
	print('\nat z = 2.0'); wget.download(link_to_emu2_file,   path_to_emul_folder)

	print('...done')
	return None