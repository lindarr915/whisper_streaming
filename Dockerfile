# File name: Dockerfile
FROM rayproject/ray-ml:2.22.0-py311-gpu

USER root
RUN apt-get update -y && apt-get install curl -y
#Add the NVIDIA key for CUDA 12 repositories
RUN curl "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb" -o cuda.deb && dpkg -i cuda.deb && rm cuda.deb

#Install CUDA 12 runtime and development packages along with libcudnn and libcublas
RUN apt-get update -y && apt-get install -y libcudnn9-cuda-12 libcublas-12-1 && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt install cuda-toolkit-12 -y

USER $RAY_UID
COPY requirements.txt . 
RUN pip install -r requirements.txt
WORKDIR /serve_app
# COPY . . 

USER root
# RUN chmod 777 /serve_app/audio_files

ENV TZ=Asia/Taipei
USER $RAY_UID
