from setuptools import setup, find_packages

def installs(file):
    """Read requirements from file, skipping comments and empty lines"""
    with open(file) as f:
        return [req.strip() for req in f.readlines()
                if req.strip() and not req.strip().startswith('#')]


setup(
    name="SimilarityText",
    version="0.3.0",
    author= "Fabio Alberti",
    author_email = "fabiocax@gmail.com",
    description="Find the similarity between two texts using AI with advanced ML and Transformer support",
    #long_description = file: README.m,
    #long_description_content_type = text/markdown,
    url = "https://github.com/fabiocax/SimilarityText",
    packages=find_packages(),
    include_package_data=True,
    install_requires=installs("requirements.txt"),
    extras_require={
        'transformers': ['sentence-transformers>=2.0.0', 'torch>=1.9.0', 'transformers>=4.0.0'],
    },
    python_requires='>=3.7',
)
