
ARG base_image=centos:7
ARG date

FROM ${base_image}

LABEL build-date=${date}

RUN yum -y update && yum clean all
RUN yum -y install centos-release-scl
RUN yum -y install devtoolset-7

RUN echo -e "\nsource /opt/rh/devtoolset-7/enable" >> /etc/skel/.bashrc

