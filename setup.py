# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='english_accents_mining',
    version='0.1.0',
    description='Data mining project on "Speech Accent Archive" dataset',
    long_description=readme,
    author='Jérémie Piotte',
    author_email='jeremie.piotte@gmail.com',
    url='https://github.com/piotte13/Speech-Accent-Mining',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

