from setuptools import setup, find_packages

def installs(file):
    return [req.strip() for req in open(file).readlines()]


setup(
    name="SimilarityText",
    version="0.1.2",
    description="Find the similarity between two texts using AI",
    packages=find_packages(),
    include_package_data=True,
    install_requires=installs("requeriments.txt"),

)