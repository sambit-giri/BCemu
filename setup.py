'''
Created on 1 June 2021
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

BCemu_link = 'https://github.com/sambit-giri/BCemu.git'

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='BCemu',
      version='1.1.4',
      description='Using emulators to implement baryonic effects.',
      url=BCemu_link,
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={'BCemu': ['input_data/*']},
      install_requires=requirements,
      # zip_safe=False,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
