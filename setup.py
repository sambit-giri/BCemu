'''
Created on 1 June 2021
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

BCMemu_link = 'https://github.com/sambit-giri/BCemu.git'

setup(name='BCemu',
      version='1.0',
      description='Using emulators to implement baryonic effects.',
      url=BCMemu_link,
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      # packages=find_packages("src"),
      # package_dir={"": "src"},
      package_dir = {'BCemu' : 'src'},
      packages=['BCemu'],
      package_data={'BCemu': ['input_data/*.rst']},
      install_requires=['numpy', 'scipy', 'matplotlib', 'astropy',
                        'scikit-learn', 'smt==1.0.0', 'cython', 'wget'],
      zip_safe=False,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
