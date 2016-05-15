'''
Created on Aug 8, 2015
@author: Jonas Wallin
'''
from Cython.Build import cythonize
from numpy import get_include

#try:
from setuptools import setup, Extension
#except ImportError:
#	try:
#		from setuptools.core import setup, Extension
#	except ImportError:
#		from distutils.core import setup, Extension

include_dirs = [get_include(),
               #  '/usr/include', '/usr/local/include',
                '/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/']
		
metadata = dict(
	  name='Mixture',
      version='0.1',
      author='Jonas Wallin',
      url='https://github.com/JonasWallin/Mixture',
      author_email='jonas.wallin81@gmail.com',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
      requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'BayesFlow',
                'scipy'],
      packages=['Mixture',
                'Mixture.Bayesflow',
                'Mixture.density',
                'Mixture.density.purepython',
                'Mixture.util'],
        ext_modules = [Extension("Mixture.util.cython_Bessel",sources=["Mixture/util/cython_Bessel.pyx", "Mixture/util/c/bessel.c"],
                       include_dirs = include_dirs,
                        #libraries=['gfortran','m','cblas','clapack'],
                       language='c')],
      package_dir={'Mixture'                   : 'Mixture/',
                   'Mixture.Bayesflow'         : 'Mixture/Bayesflow',
                   'Mixture.density'           : 'Mixture/density',
                   'Mixture.density.purepython':'Mixture/density/purepython',
                   'Mixture.util'              : 'Mixture/util'})

setup(**metadata)
