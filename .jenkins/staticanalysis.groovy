#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String buildType = debug ? 'Debug' : 'RelWithDebInfo'

    // comment

    def test_dir =  "Tensile/Tests"
    def test_marks = "unit"

    def command = """#!/usr/bin/env bash
            set -ex

            hostname

            cd ${project.paths.project_build_prefix}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`

            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            tox -v --workdir /tmp/.tensile-tox -e lint

            mkdir build
            pushd build

            popd

            doxygen docs/Doxyfile
            """

    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        throw e
    }

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/html",
                reportFiles: 'index.html',
                reportName: 'Documentation',
                reportTitles: 'Documentation'])
}


def runCI =
{
    nodeDetails, jobName ->

    def prj = new rocProject('Tensile', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = true
    boolean staticAnalysis = true

    prj.timeout.test = 30

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 6 * * 6')])]))
    stage(urlJobName) {
        runCI([ubuntu18:['any']], urlJobName)
    }
}
