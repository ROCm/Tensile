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
    nodeDetails, jobName ->

    def prj = new rocProject('Tensile', 'PreCheckin')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, false)
    }
    
    def testCommand =
    {
        platform, project->

        def test_marks = "unit or pre_checkin"
        commonGroovy.runTestCommand(platform, project, test_marks)   
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)

}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 6 * * 6')])], 
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 6 * * 6')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu16:['gfx900','gfx906','gfx908']]), 
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu16:['gfx900','gfx906','gfx908']]), 
                       "rocm-docker":([ubuntu16:['gfx900','gfx906','gfx908']])]

    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each 
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each 
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 6 * * 6')])]))
        stage(urlJobName) {
            runCI([ubuntu16:['any']], urlJobName)
        }
    }
} 
