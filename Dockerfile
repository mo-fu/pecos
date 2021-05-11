from python:3.9-slim-buster
RUN apt-get update && apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev gfortran libatlas-base-dev g++ make libomp-7-dev bc git
COPY pecos-git /pecos
WORKDIR /pecos
RUN pip install -e .
RUN mkdir /pecos_lib
RUN cp pecos/core/libpecos_float32.cpython-39-x86_64-linux-gnu.so /pecos_lib/
RUN rm -rf /pecos
