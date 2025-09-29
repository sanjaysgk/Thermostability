import os
import re
import setuptools

NAME             = "hla-thermo"
AUTHOR           = "Sanjay"
AUTHOR_EMAIL     = "sanjay dot sondekoppagopalakrishna at monash dot edu
"
DESCRIPTION      = "A pipeline for thermostability profiling of HLA-bound peptides

"
LICENSE          = "MIT"
KEYWORDS         = "thermostability, HLA, immunopeptidomics, peptide, DIA-NN, proteomics, bioinformatics, mass spectrometry, MHC, stability profiling, pipeline
"
URL              = "https://github.com/sanjaysgk/" + NAME
README           = ".github/README.md"
CLASSIFIERS      = [
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Operating System :: OS Independent",
  "Development Status :: 3 - Alpha",
  "License :: OSI Approved :: MIT License",
  "Topic :: Scientific/Engineering :: Bio-Informatics",
  "Topic :: Scientific/Engineering :: Chemistry",
  "Topic :: Scientific/Engineering :: Medical Science Apps",
  "Intended Audience :: Science/Research",
  "Intended Audience :: Healthcare Industry",
  
]
INSTALL_REQUIRES = [
  "numpy>=1.24",
  "pandas>=2.0",
  "scipy>=1.11",
  "matplotlib>=3.8",
  "typer>=0.9",
  "rich>=13.3",
  
]
ENTRY_POINTS = {
  
}
SCRIPTS = [
  
]

HERE = os.path.dirname(__file__)

def read(file):
  with open(os.path.join(HERE, file), "r") as fh:
    return fh.read()

VERSION = re.search(
  r'__version__ = [\'"]([^\'"]*)[\'"]',
  read(NAME.replace("-", "_") + "/__init__.py")
).group(1)

LONG_DESCRIPTION = read(README)

if __name__ == "__main__":
  setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    entry_points=ENTRY_POINTS,
    scripts=SCRIPTS,
    include_package_data=True    
  )
