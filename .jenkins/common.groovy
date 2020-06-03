// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String compiler = jobName.contains('hipclang') ? 'hipcc' : 'hcc'
    String cov = jobName.contains('hipclang') ? "V3" : "V2"
    String buildType = debug ? 'Debug' : 'RelWithDebInfo'
    String parallelJobs = jobName.contains('hipclang') ? "export HIPCC_COMPILE_FLAGS_APPEND=-parallel-jobs=2" : ":"

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
            tox -v --workdir /tmp/.tensile-tox -e lint
            #### temporarily enable --no-merge-files until hipclang update is posted
            tox -v --workdir /tmp/.tensile-tox -e py35 -- ${test_dir} -m "${test_marks}" --junit-xml=\$(pwd)/python_unit_tests.xml --tensile-options=--no-merge-files,--cxx-compiler=${compiler} --timing-file=\$(pwd)/timing-\$gpuArch.csv

            mkdir build
            pushd build

            export PATH=/opt/rocm/bin:$PATH
            cmake -DCMAKE_BUILD_TYPE=${buildType} -DCMAKE_CXX_COMPILER=${compiler} -DCODE_OBJECT_VERSION=${cov} -DTensile_ROOT=\$(pwd)/../Tensile ../HostLibraryTests
            make -j\$(nproc)

            popd

            doxygen docs/Doxyfile
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

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/html",
                reportFiles: 'index.html',
                reportName: 'Documentation',
                reportTitles: 'Documentation'])
}

def publishResults(project)
{
    try
    {
        archiveArtifacts "${project.paths.project_build_prefix}/timing*.csv"
    }
    finally
    {
        try
        {
            junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
        }
        finally
        {
            junit "${project.paths.project_build_prefix}/python_tests.xml"
        }
    }
}

def runTestCommand (platform, project, jobName, test_marks)
{
    def test_dir =  "Tensile/Tests"

    String compiler = jobName.contains('hipclang') ? 'hipcc' : 'hcc'

    def command = """#!/usr/bin/env bash
            set -x

            hostname

            export PATH=/opt/rocm/bin:\$PATH
            cd ${project.paths.project_build_prefix}

            gpuArch=`/opt/rocm/bin/rocm_agent_enumerator  | tail -n 1`

            pushd build
            ./TensileTests --gtest_output=xml:host_test_output.xml --gtest_color=yes
            HOST_ERR=\$?

            popd
            #### temporary fix to remedy incorrect home directory
            export HOME=/home/jenkins
            ####
            tox --version
            #### temporarily enable --no-merge-files until hipclang update is posted
            tox -v --workdir /tmp/.tensile-tox -e py35 -- ${test_dir} -m "${test_marks}" --tensile-options=--no-merge-files,--cxx-compiler=${compiler} --timing-file=\$(pwd)/timing-\$gpuArch.csv
            PY_ERR=\$?
            date

            if [[ \$HOST_ERR -ne 0 ]]
            then
                exit \$HOST_ERR
            fi

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
            publishResults(project)
        }
        catch(ee)
        {}

        throw e
    }
    publishResults(project)
}

return this
