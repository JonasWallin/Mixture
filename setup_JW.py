'''
Created on Aug 8, 2015
@author: Jonas Wallin
'''
from Cython.Build import cythonize
from numpy import get_include

try:
	from setuptools import setup, Extension
except ImportError:
	try:
		from setuptools.core import setup, Extension
	except ImportError:
		from distutils.core import setup, Extension
		
metadata = dict(
	  name='Mixture',
      version='0.1',
      author='Jonas Wallin',
      url='https://github.com/JonasWallin/Mixture',
      author_email='jonas.wallin81@gmail.com',
      requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'BayesFlow'],
      packages=['Mixture',
                'Mixture.Bayesflow'],
      package_dir={'Mixture': 'Mixture/',
                   'Mixture.Bayesflow': 'Mixture/Bayesflow'}
              )
setup(**metadata)
