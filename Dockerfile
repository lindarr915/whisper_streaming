# File name: Dockerfile
FROM rayproject/ray-ml:2.11.0-py311-gpu

RUN pip install -r requirements.txt
WORKDIR /serve_app
# COPY . . 

USER root
# RUN chmod 777 /serve_app/audio_files

ENV TZ=Asia/Taipei
USER $RAY_UID
