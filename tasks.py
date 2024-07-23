from invoke import task


@task
def hostlibtest(c, clean=False, configure=False, build=False, run=False):
    buildTo

    if clean:
        c.run("rm -rf build_hostlibtest")
    if configure:
        c.run(
            "cmake "
            "-B `pwd`/build_hostlibtest "
            "-S `pwd`/HostLibraryTests "
            "-DCMAKE_BUILD_TYPE=Debug "
            "-DCMAKE_CXX_COMPILER=amdclang++ "
            '-DCMAKE_CXX_FLAGS="-D__HIP_HCC_COMPAT_MODE__=1" '
            "-DTensile_CPU_THREADS=8 "
            "-DTensile_ROOT=`pwd`/Tensile "
            "-DTensile_VERBOSE=1"
        )
    if build:
        c.run("cmake --build `pwd`/build_hostlibtest -j4")
    if run:
        c.run("./build_hostlibtest/TensileTest")
