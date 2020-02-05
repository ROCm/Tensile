#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

properties(auxiliary.setProperties())


tensileCI:
{
    def tensile = new rocProject('Tensile', 'Debug')
    tensile.paths.build_command = 'cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=hcc -DTensile_ROOT=$(pwd)/../Tensile ../HostLibraryTests'
    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu'], tensile)

    boolean formatCheck = false

    tensile.timeout.test = 600

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/Common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    buildProject(tensile, formatCheck, nodes.dockerArray, compileCommand, null, null)

}