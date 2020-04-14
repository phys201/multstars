from setuptools import setup

setup(name='multstars',
      version='1.0',
      description='This is a package built to infer and identify multiplicity trends in the Robo-AO M-dwarf multiplicity survey through identifying distributions towards equal-mass binaries and how mass ratios vary with separation.',
      url='http://github.com/phys201/multstars',
      author='Jasmine, Claire, and Victoria',
      author_email='jasmine.gill@cfa.harvard.edu',
      license='GPLv3',
      packages=['multstars',
		'multstars.model'],
      install_requires=['numpy','matplotlib','astropy','pandas','seaborn'],
      test_suite='nose.collector',
      tests_require=['nose'])

