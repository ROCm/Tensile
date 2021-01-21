// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean buildHostLibrary=true, boolean buildTox=true)
{
    project.paths.construct_build_prefix()

    String compiler = 'hipcc'
    String pythonVersion = 'py36'
    String cov = "V3"
    String buildType = debug ? 'Debug' : 'RelWithDebInfo'
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
    """

    if (buildTox)
    {
        command += """
            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            export TENSILE_COMPILER=${compiler}
            tox -v --workdir /tmp/.tensile-tox -e ${pythonVersion} -- ${test_dir} -m "${test_marks}" --junit-xml=\$(pwd)/python_unit_tests.xml --tensile-options=--cxx-compiler=${compiler} --timing-file=\$(pwd)/timing-\$gpuArch.csv
        """
    }
    if (buildHostLibrary)
    {
        command += """
            mkdir build
            pushd build

            export PATH=/opt/rocm/bin:$PATH
            cmake -DCMAKE_BUILD_TYPE=${buildType} -DCMAKE_CXX_COMPILER=${compiler} -DCODE_OBJECT_VERSION=${cov} -DTensile_ROOT=\$(pwd)/../Tensile ../HostLibraryTests
            make -j\$(nproc)

            popd
            """
    }

    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        try
        {
            if (buildTox)
            {
                junit "${project.paths.project_build_prefix}/python_unit_tests.xml"
            }
        }
        catch(ee)
        {}

        throw e
    }

    junit "${project.paths.project_build_prefix}/python_unit_tests.xml"
}

def publishResults(project, boolean runHostTest=true)
{
    try
    {
        archiveArtifacts "${project.paths.project_build_prefix}/timing*.csv"
    }
    finally
    {
        try
        {
            if (runHostTest) junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
        }
        finally
        {
            junit "${project.paths.project_build_prefix}/python_tests.xml"
        }
    }
}

def runTestCommand (platform, project, jobName, test_filter, boolean runHostTest=true, boolean runToxTest=true)
{
    def test_dir =  "Tensile/Tests"
    String compiler = 'hipcc'
    String pythonVersion = 'py36'

    def command = """#!/usr/bin/env bash
            set -x

            hostname

            export PATH=/opt/rocm/bin:\$PATH
            cd ${project.paths.project_build_prefix}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`
            """

    if (runHostTest)
    {
        command += """
            pushd build
            ./TensileTests --gtest_output=xml:host_test_output.xml --gtest_color=yes
            HOST_ERR=\$?
            popd
            
            if [[ \$HOST_ERR -ne 0 ]]
            then
                exit \$HOST_ERR
            fi
        """
    }

    if (runToxTest)
    {
        command += """
            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            export TENSILE_COMPILER=${compiler}
            tox -v --workdir /tmp/.tensile-tox -e ${pythonVersion} -- ${test_dir} -m "${test_filter}" --tensile-options=--cxx-compiler=${compiler} --timing-file=\$(pwd)/timing-\$gpuArch.csv
            PY_ERR=\$?
            date

            if [[ \$PY_ERR -ne 0 ]]
            then
                exit \$PY_ERR
            fi
        """
    }

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
            publishResults(project, runHostTest)
        }
        catch(ee)
        {}

        throw e
    }
    publishResults(project, runHostTest)
}

return this
