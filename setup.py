#!/usr/bin/env python
from setuptools import setup


__version__="0.1a"

setup(name='asr',
      version=__version__,
      description='Models for a small project in the field of speech recognition',
      author='Sebastian Mueller',
      author_email='semueller@techfak.uni-bielefeld.de',
      packages=['asr'],
      url='http://github.com/semueller',
      license='GPL3',
      install_requires=['torch>=1.0.1'],
)

