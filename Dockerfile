# File name: Dockerfile
FROM rayproject/ray-ml:2.11.0-py311-gpu

USER root
RUN apt-get update &&  apt install cuda-toolkit-12 -y

USER $RAY_UID
COPY requirements.txt . 
RUN pip install -r requirements.txt
WORKDIR /serve_app
# COPY . . 

USER root
# RUN chmod 777 /serve_app/audio_files

ENV TZ=Asia/Taipei
USER $RAY_UID
