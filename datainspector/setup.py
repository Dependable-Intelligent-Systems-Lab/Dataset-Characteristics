import setuptools
from distutils.core import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
  name = 'dace',         
  packages = ['dace'],   
  version = '0.0.1',     
  license='MIT',       
  description = 'D-ACE: Dataset Assessment and Characteristics Evaluation',
  author = 'Naveed Akram',               
  author_email = 'naveed.akram@iese.fraunhofer.de',     
  url = 'https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics',   
  download_url = 'https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics',   
  keywords = ['dataset', 'aisafety', 'safety', 'trustworthyai'], 
  setup_requires = ['wheel'],
  install_requires=[            # I get to this in a second
          'pandas',
          'scikit-learn',
          'scikit-plot',
          'seaborn',
          'scikit-dimension',
          '',
          'pingouin',
          'graphviz',
          'numba==0.53',
          'safeml',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.10',
  ],
)
