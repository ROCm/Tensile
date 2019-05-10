#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@akilajenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
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

TensileCI:
{
    def tensile = new rocProject('Tensile')
    // customize for project
    tensile.paths.build_command = cmake -D CMAKE_BUILD_TYPE=Debug ../lib && make -j16
    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx906'], tensile)

    boolean formatCheck = false
    
    try
    {
        timeout(time: 1, unit: 'HOURS')
        {
            def compileCommand =
            {
                platform, project->

                project.paths.construct_build_prefix()
                def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}
                        mkdir build && cd build
                        export PATH=/opt/rocm/bin:$PATH
                        ${project.paths.build_command}
                        ./test/TensileTests --gtest_output=xml:host_test_output.xml --gtest_color=yes
                        """

                platform.runCommand(this, command)
            }
        }
    }
    finally
    {
        junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
    }
   
    def test_dir = isJobStartedByTimer() ? "Tensile/Tests/nightly" : "Tensile/Tests/pre_checkin"
    try
    {
        timeout(time: 3, unit: 'HOURS')
        {
            def testCommand =
            {
                platform, project->

                def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}
                        tox --version
                        tox -vv --workdir /tmp/.tensile-tox Tensile/UnitTests ${test_dir} -e lint
                        tox -vv --workdir /tmp/.tensile-tox Tensile/UnitTests ${test_dir} -e py27
                        tox -vv --workdir /tmp/.tensile-tox Tensile/UnitTests ${test_dir} -e py35
                        """

                platform.runCommand(this, command)
            }
        }
    }
    finally
    {
        junit "${project.paths.project_build_prefix}/*_tests.xml"
    }
    
    def packageCommand =
    {
        platform, project->

        def command = """
                      set -x
                      cd ${project.paths.project_build_prefix}/build
                      make package
                      rm -rf package && mkdir -p package
                      mv *.deb package/
                      dpkg -c package/*.deb
                      """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.deb""")
    }

    buildProject(tensile, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
    
}