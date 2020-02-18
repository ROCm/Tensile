
ARG base_image=tensile/run-centos-7
ARG date

FROM ${base_image}

LABEL build-date=${date}

RUN yum -y install llvm-toolset-7-llvm-devel \
                   llvm-toolset-7-llvm-static \
                   llvm-toolset-7-cmake \
                   git ncurses-devel zlib-devel

RUN echo -e "\nsource /opt/rh/llvm-toolset-7/enable" >> /etc/skel/.bashrc

