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
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String compiler = 'hipcc'
    String pythonVersion = 'py3'
    // Do release build of HostLibraryTests on CI until it is upgraded to rocm 5.3 to
    // avoid bug causing long build times of certain files.
    String buildType = 'Release' // debug ? 'Debug' : 'RelWithDebInfo'
    String parallelJobs = "export HIPCC_COMPILE_FLAGS_APPEND=-parallel-jobs=2"

    // comment

    def test_dir =  "Tensile/Tests"
    def test_marks = "unit"

    def command = """#!/usr/bin/env bash
            set -ex

            hostname

            cd ${project.paths.project_build_prefix}
            ${parallelJobs}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`

            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            export TENSILE_COMPILER=${compiler}
            tox -v --workdir /tmp/.tensile-tox -e ${pythonVersion} -- ${test_dir} -m "${test_marks}" --junit-xml=\$(pwd)/python_unit_tests.xml --timing-file=\$(pwd)/timing-\$gpuArch.csv

            mkdir build
            pushd build

            export PATH=/opt/rocm/bin:$PATH
            cmake -DCMAKE_BUILD_TYPE=${buildType} -DCMAKE_CXX_COMPILER=${compiler} -DTensile_ROOT=\$(pwd)/../Tensile ../HostLibraryTests
            make -j\$(nproc)

            popd
            """

    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        try
        {
            junit "${project.paths.project_build_prefix}/python_unit_tests.xml"
        }
        catch(ee)
        {}

        throw e
    }

    junit "${project.paths.project_build_prefix}/python_unit_tests.xml"
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
}

def runTestCommand (platform, project, jobName, test_marks, boolean skipHostTest=false)
{
    def test_dir =  "Tensile/Tests"

    String compiler = 'hipcc'
    String pythonVersion = 'py3'
    String markSkipHostTest = skipHostTest ? "#" : ""
    String markSkipExtendedTest = !test_marks.contains("extended") ? "--gtest_filter=-\"*Extended*\"" : ""

    def command = """#!/usr/bin/env bash
            set -x

            hostname

            export PATH=/opt/rocm/bin:\$PATH
            cd ${project.paths.project_build_prefix}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`

            ${markSkipHostTest}pushd build
            ${markSkipHostTest}./TensileTests ${markSkipExtendedTest} --gtest_output=xml:host_test_output.xml --gtest_color=yes
            ${markSkipHostTest}HOST_ERR=\$?
            ${markSkipHostTest}popd

            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            export TENSILE_COMPILER=${compiler}
            tox -v --workdir /tmp/.tensile-tox -e ${pythonVersion} -- ${test_dir} -m "${test_marks}" --timing-file=\$(pwd)/timing-\$gpuArch.csv
            PY_ERR=\$?
            date

            ${markSkipHostTest}if [[ \$HOST_ERR -ne 0 ]]
            ${markSkipHostTest}then
            ${markSkipHostTest}    exit \$HOST_ERR
            ${markSkipHostTest}fi

            if [[ \$PY_ERR -ne 0 ]]
            then
                exit \$PY_ERR
            fi
        """

    // This awkward sequence prevents an exception in runCommand() from being
    // eaten by an exception in publishResults(), while allowing partial results
    // to still be published.
    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        try
        {
            publishResults(project, skipHostTest)
        }
        catch(ee)
        {}

        throw e
    }
    publishResults(project, skipHostTest)
}

return this
