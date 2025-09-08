import os, tqdm
import wget, json
# import pkg_resources
# from importlib.resources import files

def get_package_resource_path(package_name, resource_path):
    """
    Get the path to a resource file within a package, with automatic fallback
    for older Python versions.
    
    Parameters
    ----------
    package_name : str
        Name of the package (e.g., 'tools21cm')
    resource_path : str
        Path to the resource relative to the package root
        (e.g., 'input_data/central_geographic_position.txt')
    
    Returns
    -------
    str or Path
        Path to the resource file that can be used with file reading functions
    
    Examples
    --------
    >>> path = get_package_resource_path('tools21cm', 'input_data/central_geographic_position.txt')
    >>> data = np.loadtxt(path)
    """
    
    # Try importlib.resources first (Python 3.9+)
    try:
        from importlib.resources import files
        resource_file = files(package_name) / resource_path
        return str(resource_file)
    
    except ImportError:
        # Fall back to importlib_resources (Python 3.6-3.8)
        try:
            from importlib_resources import files
            resource_file = files(package_name) / resource_path
            return str(resource_file)
        
        except ImportError:
            # Fall back to pkg_resources (older versions)
            try:
                import pkg_resources
                return pkg_resources.resource_filename(package_name, resource_path)
            
            except ImportError:
                # Last resort: try relative path (development mode)
                import os
                from pathlib import Path
                try:
                    # Get the package's location
                    package = __import__(package_name)
                    package_dir = Path(package.__file__).parent
                    resource_file = package_dir / resource_path
                    
                    if resource_file.exists():
                        return str(resource_file)
                    else:
                        raise FileNotFoundError(f"Resource not found: {resource_path}")
                        
                except Exception as e:
                    raise ImportError(
                        f"Could not locate resource '{resource_path}' in package '{package_name}'. "
                        f"Please install 'importlib_resources' for Python < 3.9 or ensure the package is properly installed."
                    ) from e
                
def _download_file(url, path, force=False):
    """Helper function to download a file, with an option to overwrite."""
    if os.path.exists(path) and not force:
        print(f"  Exists: {os.path.basename(path)}")
        return
    elif os.path.exists(path) and force:
        print(f"  Overwriting: {os.path.basename(path)}")
        os.remove(path) # Remove the old file before downloading
    else:
        print(f"  Downloading: {os.path.basename(path)}")
    
    try:
        wget.download(url, path, bar=None) 
    except Exception as e:
        print(f"\n  ERROR downloading {url}. Error: {e}")

def download_emulators(model_name='BCemu2021', force_download=False):
    """
    Downloads the specified emulator model files.

    Args:
        model_name (str): The name of the model to download. 
                          Either 'BCemu2021' or 'BCemu2025'.
        force_download (bool): If True, existing emulator files will be deleted
                               and re-downloaded. Defaults to False.
    """
    print(f"Attempting to download '{model_name}' emulators...")
    if force_download:
        print("Force download is ON. Existing files will be overwritten.")

    package_name = "BCemu"
    data_dir_name = "input_data"
    path_to_emul_folder = get_package_resource_path(package_name, data_dir_name)
    os.makedirs(path_to_emul_folder, exist_ok=True)

    if model_name.lower() == 'bcemu2021':
        print('Target directory:', path_to_emul_folder)
        base_url = 'https://github.com/sambit-giri/BCemu/releases/download/v1.0/'
        files_to_download = [
            'kpls_emulator_z0_nComp3.pkl',
            'kpls_emulator_z0p5_nComp3.pkl',
            'kpls_emulator_z1_nComp3.pkl',
            'kpls_emulator_z1p5_nComp3.pkl',
            'kpls_emulator_z2_nComp3.pkl'
        ]

        for filename in files_to_download:
            url = os.path.join(base_url, filename)
            path = os.path.join(path_to_emul_folder, filename)
            _download_file(url, path, force=force_download)
        
        print('...BCemu2021 download process finished.')

    elif model_name.lower() == 'bcemu2025':
        print('Target directory:', path_to_emul_folder)
        parent_url = 'https://ttt.astro.su.se/~sgiri/data/BCemu_models/trained_emulators/'
        
        # Metadata
        print('\n----- Metadata -----')
        meta_filename = 'BCemu2025_meta.json'
        link_to_metadata = f'{parent_url}{meta_filename}'
        path_to_metadata = os.path.join(path_to_emul_folder, meta_filename)
        _download_file(link_to_metadata, path_to_metadata, force=force_download)
        
        try:
            with open(path_to_metadata, 'r') as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("ERROR: Could not read metadata file. Aborting model download.")
            return

        # Trained emulators based on metadata
        print('\n----- Model Files -----')
        for z in meta['z_grid']:
            for q2 in meta['q2_grid']:
                print(f'Processing z={z:.2f}, q2={q2:.2f}:')
                # Updated filenames for the JAX-native, version-agnostic format
                filenames = [
                    f'BCemu2025_arch_z{z:.2f}_q2_{q2:.2f}.json',
                    f'BCemu2025_emulator_z{z:.2f}_q2_{q2:.2f}.msgpack',
                    f'BCemu2025_transforms_z{z:.2f}_q2_{q2:.2f}.npz'
                ]
                for filename in filenames:
                    url = f'{parent_url}{filename}'
                    path = os.path.join(path_to_emul_folder, filename)
                    _download_file(url, path, force=force_download)
        
        print('...BCemu2025 download process finished.')

    else:
        print(f"Model '{model_name}' not recognized. Please choose 'BCemu2021' or 'BCemu2025'.")
        print('No emulators downloaded.')
    
    return None