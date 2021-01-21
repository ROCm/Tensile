#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, runHostTest, runToxTest ->

    def prj = new rocProject('Tensile', 'PreCheckin')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    prj.timeout.test = 120

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, runHostTest, runToxTest)
    }

    def testCommand =
    {
        platform, project->

        def test_filter = "pre_checkin"
        commonGroovy.runTestCommand(platform, project, jobName, test_filter, runHostTest, runToxTest)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900','gfx906','gfx908']])]

    // jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        /*if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName, false, true)
            }*/
        if (urlJobName == "compute-rocm-dkms-no-npi-hipclang")
            stage(jobName) {
                runCI(([centos7:['gfx900']]), jobName, true, false)
            }
    }

    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 6 * * 6')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['any']], urlJobName, true, true)
        }
    }
}
