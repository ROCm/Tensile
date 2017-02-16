from setuptools import setup, find_packages


setup(
    name="Tensile",
    version="2.0",
    description="An auto-tuning tool for GEMMs and higher-dimensional tensor contractions on GPUs.",
    url="https://github.com/RadeonOpenCompute/Tensile",
    author="Advanced Micro Devices",
    license="MIT",
    install_requires=["pyyaml"],
    packages=["tensile"],
    package_data={"tensile": ["tensile/*", "Source/*"]},
    #data_files=["Source", ["Client.cpp"{"tensile": ["Tensile/*"]},
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": [
        "tensile = tensile.Tensile:Tensile",
        "tensile-library-writer = tensile.TensileLibraryWriter"
        ]}
    )
