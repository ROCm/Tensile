/*******************************************************************************
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// This file is for internal AMD use.
// If you are interested in running your own Jenkins,
// please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String compiler = '/opt/rocm/bin/amdclang++'
    // Do release build of HostLibraryTests on CI until it is upgraded to rocm 5.3 to
    // avoid bug causing long build times of certain files.
    String buildType = 'Release' // debug ? 'Debug' : 'RelWithDebInfo'
    
    int systemCPUs = sh(script: 'nproc', returnStdout: true ).trim().toInteger()
    long containerRAMbytes = sh(script: 'if [ -f /sys/fs/cgroup/memory.max ]; then cat /sys/fs/cgroup/memory.max; else cat /sys/fs/cgroup/memory/memory.limit_in_bytes; fi', returnStdout: true ).trim().toLong()
    int containerRAM = containerRAMbytes / (1024 * 1024)
    int maxThreads = containerRAM / 8
    if (maxThreads > systemCPUs) maxThreads = systemCPUs
    if (maxThreads > 64) maxThreads = 64
    if (maxThreads < 1) maxThreads = 1
    
    String buildThreads = maxThreads.toString() // if hipcc is used may be multiplied by parallel-jobs

    def command = """#!/usr/bin/env bash
            set -ex
            hostname
            cd ${project.paths.project_build_prefix}

            export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4'
            export HOME=/home/jenkins
            export TENSILE_COMPILER=${compiler}
            export PATH=/opt/rocm/bin:\$PATH

            mkdir build && pushd build

            cmake ../HostLibraryTests \
                -DCMAKE_BUILD_TYPE=${buildType} \
                -DCMAKE_CXX_COMPILER=${compiler} \
                -DCMAKE_CXX_FLAGS="-D__HIP_HCC_COMPAT_MODE__=1" \
                -DTensile_CPU_THREADS=${buildThreads} \
                -DTensile_ROOT=\$(pwd)/../Tensile
            
            NPROC_BUILD=16
            if [ `nproc` -lt 16 ]
            then
              NPROC_BUILD=`nproc`
            fi
            make -j\$NPROC_BUILD

            popd
            """

    platform.runCommand(this, command)
}

def publishResults(project, boolean skipHostTest=false)
{
    try
    {
        archiveArtifacts "${project.paths.project_build_prefix}/timing*.csv"
    }
    finally
    {
        try
        {
            if (!skipHostTest) junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
        }
        finally
        {
            junit "${project.paths.project_build_prefix}/python_tests.xml"
        }
    }
    try
    {
        archiveArtifacts "${project.paths.project_build_prefix}/htmlcov/**/*"
    }
    finally
    {
        println("Couldn't find coverage artifacts")
    }
}

def runTestCommand (platform, project, jobName, testMark, boolean skipHostTest=false, boolean skipUnitTest=false)
{
    String compiler = '/opt/rocm/bin/amdclang++'
    String markSkipExtendedTest = !testMark.contains("extended") ? "\"--gtest_filter=-*Extended*:*Ocl*\"" : "\"--gtest_filter=-*Ocl*\""

    def command = """#!/usr/bin/env bash
            check_err() {
              ERR=\$?; [ \$ERR -ne 0 ] && exit \$ERR
            }

            set -x
            hostname
            date
            cd ${project.paths.project_build_prefix}

            export PATH=/opt/rocm/bin:\$PATH
            export HOME=/home/jenkins
            export TENSILE_COMPILER=${compiler}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`

            if ! ${skipHostTest}; then
              pushd build
              ./TensileTests ${markSkipExtendedTest} --gtest_output=xml:host_test_output.xml --gtest_color=yes
              check_err
              popd
            fi

            tox --version

            tox run -e ci -- -m ${testMark} --timing-file=\$(pwd)/timing-\$gpuArch.csv
            check_err

            if ! ${skipUnitTest}; then 
              tox run -e unittest -- --cov-report=xml:cobertura.xml
              check_err
            fi
        """

    // This awkward sequence prevents an exception in runCommand() from being
    // eaten by an exception in publishResults(), while allowing partial results
    // to still be published.


    platform.runCommand(this, command)
    // try {
    //     platform.runCommand(this, command)
    // }
    // catch(e) {
    //     try {
    //         publishResults(project, skipHostTest)
    //     }
    //     catch(ee) {}
    //     throw e
    // }
    archiveArtifacts "${project.paths.project_build_prefix}/timing*.csv"
    if (!skipHostTest) junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
    junit "${project.paths.project_build_prefix}/python_tests.xml"
    recordCoverage(tools: [[parser: 'COBERTURA']])
    // publishResults(project, skipHostTest)
}

return this
