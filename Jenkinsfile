#!/usr/bin/env groovy

/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

parallel rocm_fiji: {

  currentBuild.result = "SUCCESS"
  node('rocm-1.3 && fiji')
  {
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"

    // Run the containers preperation script
    // Note, exported environment variables do not live outside of sh step
    sh ". /home/jenkins/prep-env.sh"

    // The following try block performs build steps
    try
    {
      dir("${scm_dir}") {
        stage("Clone") {
          checkout scm
        }
      }

      withEnv(["PATH=${PATH}:/opt/rocm/bin"]) {

        // Record important versions of software layers we use
        sh '''clang++ --version
              cmake --version
              hcc --version
              hipconfig --version
        '''

        dir("${build_dir_release}") {
          stage("configure clang release") {
            // withEnv(['CXXFLAGS=-I /usr/include/c++/4.8 -I /usr/include/x86_64-linux-gnu/c++/4.8  -I /usr/include/x86_64-linux-gnu', 'HIP_PATH=/opt/rocm/hip']) {
            // --amdgpu-target=AMD:AMDGPU:8:0:3
              sh "cmake -DCMAKE_BUILD_TYPE=Release ${scm_dir}"
            // }
          }

          stage("Build") {
            // withEnv(['HCC_AMDGPU_TARGET=AMD:AMDGPU:7:0:1,AMD:AMDGPU:8:0:3']) {
              sh 'make -j 8'
            // }
          }

          stage("unit tests") {
            // To trim test time, only execute single digit tests
            sh "python ${scm_dir}/test/test.py -b HIP --problem-set full"
          }

        }
      }
    }
    catch( err )
    {
        currentBuild.result = "FAILURE"

        def email_list = emailextrecipients([
                [$class: 'CulpritsRecipientProvider']
        ])

        // CulpritsRecipientProvider below doesn't help, because nobody uses their real email address
        // emailext  to: "kent.knox@amd.com", recipientProviders: [[$class: 'CulpritsRecipientProvider']],
        //       subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
        //       body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

        // Disable email for now
        mail  to: "kent.knox@amd.com, david.tanner@amd.com, tingxing.dong@amd.com",
              subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
              body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

        throw err
    }
  }
}, catalyst: {
  node ('opencl && hawaii'){
    stage('Checkout') {
      env.CXXFLAGS = "-Werror"
      checkout scm
    }
    stage('Clang') {
      sh '''
          CXX='clang++-3.8' python test/test.py -b cl --no-validate --problem-set full
      '''
    }
    stage('GCC') {
      sh '''
          CXX='g++-4.8' python test/test.py -b cl --no-validate --problem-set full 
      '''
    }
  }
}
