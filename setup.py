from setuptools import setup, find_packages

setup(
    name='2DTMD_ANML',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.26.4',
        'matplotlib==3.8.3',
        'scipy==1.12.0',
        'pandas==2.2.1',
        'seaborn==0.13.2',
        'lmfit==1.2.2',
        'peakutils==1.3.4',
        'openpyxl==3.1.2',
        'joblib==1.3.2',
        'scikit-learn==1.5.1',
        'threadpoolctl==3.3.0',
        'tensorflow==2.16.2',
    ],
    description='Data analysis tools for characterization techniques',
    author='Eridani Rojas',
    author_email='eridanirojas@u.boisestate.edu',
    url='https://github.com/eridanirojas/2DTMD_ANML',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

