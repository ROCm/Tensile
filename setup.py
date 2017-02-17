from setuptools import setup


setup(
    name="Tensile",
    version="2.0",
    description="An auto-tuning tool for GEMMs and higher-dimensional tensor contractions on GPUs.",
    url="https://github.com/RadeonOpenCompute/Tensile",
    author="Advanced Micro Devices",
    license="MIT",
    install_requires=["pyyaml"],
    packages=["Tensile"],
    data_files=[
        ("Source", [
            "Source/BenchmarkClient.cmake",
            "Source/CMakeLists.txt",
            "Source/FindHCC.cmake",
            "Source/MathTemplates.cpp",
            "Source/SetupTeardown.cpp",
            "Source/TensileTypes.h",
            "Source/Client.cpp",
            "Source/FindHIP.cmake",
            "Source/MathTemplates.h",
            "Source/SolutionHelper.cpp",
            "Source/TensileCreateLibrary.cmake",
            "Source/Tools.cpp",
            "Source/Client.h",
            "Source/EnableWarnings.cmake",
            "Source/FindOpenCL.cmake",
            "Source/ReferenceCPU.h",
            "Source/SolutionHelper.h",
            "Source/Tools.h",
            ] ),
        ("Configs", [
            "Configs/jenkins_sgemm_defaults.yaml",
            "Configs/jenkins_dgemm_defaults.yaml",
            "Configs/rocblas_sgemm.yaml",
            "Configs/rocblas_dgemm.yaml",
            "Configs/rocblas_cgemm.yaml",
            "Configs/sgemm_5760.yaml",
            ] ),
        ], 
    include_package_data=True,
    #zip_safe=False,
    entry_points={"console_scripts": [
        "tensile = Tensile.Tensile:Tensile",
        "tensile_library_writer = Tensile.TensileCreateLibrary:TensileCreateLibrary"
        ]}
    )
