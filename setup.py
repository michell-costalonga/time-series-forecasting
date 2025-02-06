from setuptools import setup, find_packages

setup(
    name="time_series_forecasting",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Dependências necessárias para o pacote funcionar
        "numpy",
        "pandas",
        "prophet",
        "datetime",
    ],
    author="Michell Costalonga",
    author_email="michell.costalonga@picpay.com",
    description="Um pacote para a previsão de séries temporais com entrada e saída de um PySpark Dataframe",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/michell-costalonga/time-series-forecasting",  # Caso publique no GitHub
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
