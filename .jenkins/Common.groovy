// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
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
            tox -v --workdir /tmp/.tensile-tox -e lint
            """

    platform.runCommand(this, command)

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/html",
                reportFiles: 'index.html',
                reportName: 'Documentation',
                reportTitles: 'Documentation'])
}

def runTestCommand (platform, project, test_marks)
{
    def test_dir =  "Tensile/Tests"

    try
    {
        def command = """#!/usr/bin/env bash
                set -x

                hostname

                export PATH=/opt/rocm/bin:\$PATH
                cd ${project.paths.project_build_prefix}

                pushd build
                ./TensileTests --gtest_output=xml:host_test_output.xml --gtest_color=yes
                HOST_ERR=\$?

                popd
                tox --version
                tox -v --workdir /tmp/.tensile-tox -e py35 -- ${test_dir} -m "${test_marks}"
                PY_ERR=$?
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
        platform.runCommand(this, command)
    }
    finally
    {
        junit "${project.paths.project_build_prefix}/build/host_test_output.xml"
        junit "${project.paths.project_build_prefix}/*_tests.xml"
    }
}

return this