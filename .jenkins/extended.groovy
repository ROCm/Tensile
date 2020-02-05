#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName->

    def tensile = new rocProject('Tensile', 'Extended')
    tensile.paths.build_command = 'cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_CXX_COMPILER=hcc -DTensile_ROOT=$(pwd)/../Tensile ../HostLibraryTests'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    tensile.timeout.test = 600

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->

        def test_marks = "unit or pre_checkin or extended"
        commonGroovy.runTestCommand(platform, project, test_marks)
    }

    buildProject(tensile, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])],
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    Set standardJobNameSet = ["compute-rocm-dkms-no-npi", "compute-rocm-dkms-no-npi-hipclang", "rocm-docker"]

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu16:['gfx900'],ubuntu16:['gfx906'],ubuntu16:['gfx908']]),
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu16:['gfx900'],ubuntu16:['gfx906'],ubuntu16:['gfx908']]),
                       "rocm-docker":([ubuntu16:['gfx900'],ubuntu16:['gfx906'],ubuntu16:['gfx908']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    Set seenJobNames = []
    jobNameList.each
    {
        jobName, nodeDetails->
        seenJobNames.add(jobName)
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    if(!seenJobNames.contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        runCI([ubuntu16:['gfx906']], urlJobName)
    }
}
