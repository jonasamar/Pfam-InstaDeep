import setuptools

setuptools.setup(
    name='pfam',
    version='0.0.1',
    author='Jonas Amar',
    author_email='jonas.amar@etu.minesparis.psl.eu',
    description='pfam libraries',
    packages=['DeepLibrary'],
    install_requires=[
        'matplotlib',
        "scikit-learn",
        "seaborn==0.12.2",
        "pandas",
        "numpy",
        "statsmodels",
        "openpyxl",
        "scipy",
        "tensorflow",
        "tokenizer",
        "transformers",
        "torch",
        "tensorflow",
        "joblib",
        "tqdm",
        "xgboost"
    ]
)
