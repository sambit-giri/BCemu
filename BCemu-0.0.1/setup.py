'''
Created on 1 June 2021
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

BCMemu_link = 'https://github.com/sambit-giri/BCemu.git'

setup(name='BCemu',
      version='0.0.1',
      description='Using emulators to implement baryonic effects.',
      url=BCMemu_link,
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      package_dir = {'BCMemu' : 'src'},
      packages=['BCMemu'],
      package_data={'BCMemu': ['input_data/*.pkl']},
      install_requires=['numpy', 'scipy', 'matplotlib', 'astropy',
                        'scikit-learn', 'smt', 'cython'],
      zip_safe=False,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
