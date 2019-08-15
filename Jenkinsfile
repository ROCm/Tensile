#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 3 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

tensileCI:
{
    def tensile = new rocProject('Tensile')
    tensile.paths.build_command = 'cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=hcc -DCMAKE_CXX_FLAGS=-Werror -DTensile_ROOT=$(pwd)/../Tensile ../HostLibraryTests'
    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900','gfx906'], tensile)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->
        try
        {
            project.paths.construct_build_prefix()

            def command = """#!/usr/bin/env bash
                    set -ex

                    hostname

                    export PATH=/opt/rocm/bin:$PATH
                    cd ${project.paths.project_build_prefix}

                    mkdir build
                    pushd build
                    ${project.paths.build_command}
                    make -j

                    popd
                    tox --version
                    tox -vv --workdir /tmp/.tensile-tox -e lint
                    """

            platform.runCommand(this, command)
        }
        finally
        {
        }
    }

    tensile.timeout.test = 10

    def test_dir = auxiliary.isJobStartedByTimer() ? "Tensile/Tests/nightly" : "Tensile/Tests/pre_checkin"
    def testCommand =
    {
        platform, project->
        try
        {
            def command = """#!/usr/bin/env bash
                    set -ex

                    hostname

                    export PATH=/opt/rocm/bin:$PATH
                    cd ${project.paths.project_build_prefix}

                    pushd build
                    ./TensileTests --gtest_output=xml:host_test_output.xml --gtest_color=yes

                    popd
                    tox --version
                    tox -vv --workdir /tmp/.tensile-tox -e py35 -- Tensile/UnitTests ${test_dir}
                    """
            platform.runCommand(this, command)
        }
        finally
        {
            junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
            junit "${project.paths.project_build_prefix}/*_tests.xml"
        }
    }
    def packageCommand = null

    buildProject(tensile, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
