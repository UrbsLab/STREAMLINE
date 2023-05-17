# TODO: Conform this is correct
from setuptools import setup

setup(
    name='streamline',
    version='0.2.6',
    description='Simple Transparent End-To-End Automated Machine Learning '
                'Pipeline for Supervised Learning in Tabular Binary Classification Data',
    url='https://github.com/raptor419/STREAMLINE_Dev/',
    author='Harsh Bandhey',
    author_email='harsh.bandhey@cshs.org',
    license='GPL3',
    packages=[
        'streamline'
        'streamline.dataprep',
        'streamline.featurefns',
        'streamline.modelling',
        'streamline.models',
        'streamline.postanalysis',
        'streamline.utils',
              ],
    install_requires=[
        'mpi4py>=2.0',
        'matplotlib',
        'numpy',
        'optuna',
        'pandas',
        'pip',
        'pycodestyle',
        'scikit-learn',
        'scipy',
        'seaborn',
        'skrebate==0.7',
        'Sphinx',
        'sphinx-rtd-theme',
        'tqdm',
        'wheel',
        'pytest',
        'xgboost',
        'lightgbm',
        'catboost',
        'gplearn',
        'ipython',
        'fpdf',
        'scikit-XCS',
        'scikit-ExSTraCS',
        'scikit-eLCS',
                      ],

    classifiers=[
        'Development Status :: 2 - Restructuring',
        'Intended Audience :: Science/Research',
        'License :: GPL 3 License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.9',
    ],
)
