from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='letsroms',
      version='0.1.0a',
      description='Python module to sample outputs from the Regional Ocean Modeling System (ROMS) simulating a ship survey.',
      url='https://github.com/apaloczy/LeTSROMS',
      license='MIT',
      packages=['letsroms'],
      install_requires=[
          'numpy',
          'matplotlib',
          'netCDF4',
          'cartopy',
          'cmocean',
          'gsw',
          'pandas',
          'xarray',
          'stripack',
          'pyroms',
          'pygeodesy'
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
