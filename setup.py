from setuptools import setup, find_packages

def installs(file):
    return [req.strip() for req in open(file).readlines()]


setup(
    name="SimilarityText",
    version="0.1.6",
    author= "Fabio Alberti",
    author_email = "fabiocax@gmail.com",
    description="Find the similarity between two texts using AI",
    #long_description = file: README.m,
    #long_description_content_type = text/markdown,
    url = "https://github.com/fabiocax/SimilarityText",
    packages=find_packages(),
    include_package_data=True,
    install_requires=installs("requeriments.txt"),

)
