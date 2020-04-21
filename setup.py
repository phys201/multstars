from setuptools import setup

setup(name='multstars',
      version='1.0',
      description='A package built to infer and identify trends in the projected semi-major axes of binaries in the Robo-AO M-dwarf multiplicity survey.',
      url='http://github.com/phys201/multstars',
      author='Jasmine, Claire, and Victoria',
      author_email='jasmine.gill@cfa.harvard.edu',
      license='GPLv3',
      install_requires=['numpy','matplotlib','astropy','pandas','seaborn','pymc3'],
      test_suite='nose.collector',
      tests_require=['nosetests'])

