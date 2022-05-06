import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_fac",
    version="0.2.0",
    author="Marmoret Axel,Jeremy Cohen",
    author_email="axel.marmoret@irisa.fr,jeremy.cohen@cnrs.fr",
    description="Nonnegative factorization toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7"
    ],
    license='BSD',
    install_requires=[
        'nimfa',
        'numpy >= 1.18.0',
        'scipy >= 0.13.0',
        'tensorly >= 0.4.5',
    ],
    python_requires='>=3.7',
)