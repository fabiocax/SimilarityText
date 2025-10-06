from setuptools import setup, find_packages
import os

def read_file(filename):
    """Read a file and return its contents"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, encoding='utf-8') as f:
        return f.read()

def get_requirements(filename):
    """Read requirements from file, skipping comments and empty lines"""
    content = read_file(filename)
    return [
        line.strip() for line in content.splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]

# Read the long description from README
long_description = read_file('README.md')

setup(
    name="similaritytext",
    version="0.3.1",
    author="Fabio Alberti",
    author_email="fabiocax@gmail.com",
    description="Advanced text similarity and classification using AI and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabiocax/SimilarityText",
    project_urls={
        "Bug Reports": "https://github.com/fabiocax/SimilarityText/issues",
        "Source": "https://github.com/fabiocax/SimilarityText",
        "Documentation": "https://github.com/fabiocax/SimilarityText/blob/main/README.md",
        "Changelog": "https://github.com/fabiocax/SimilarityText/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'example', 'example.*']),
    include_package_data=True,
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        'transformers': [
            'sentence-transformers>=2.0.0',
            'torch>=1.9.0',
            'transformers>=4.0.0'
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        # Development Status
        'Development Status :: 4 - Beta',

        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',

        # Topic
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # License
        'License :: OSI Approved :: MIT License',

        # Programming Language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        # Operating System
        'Operating System :: OS Independent',

        # Natural Language
        'Natural Language :: English',
        'Natural Language :: Portuguese',
        'Natural Language :: Spanish',
        'Natural Language :: French',
        'Natural Language :: German',

        # Framework
        'Framework :: Jupyter',
    ],
    keywords=[
        'nlp',
        'natural language processing',
        'text similarity',
        'semantic similarity',
        'text classification',
        'machine learning',
        'deep learning',
        'transformers',
        'bert',
        'sentence transformers',
        'tf-idf',
        'cosine similarity',
        'sentiment analysis',
        'multilingual',
        'ai',
        'artificial intelligence',
    ],
    license='MIT',
)
