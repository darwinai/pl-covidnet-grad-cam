FROM fnndsc/ubuntu-python3:18.04

LABEL org.opencontainers.image.authors="DarwinAI <support@darwinai.com>, Matthew Wang <matthew.wang@darwinai.ca>"

ENV APPROOT="/usr/src/grad_cam"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR $APPROOT

COPY ["apt-requirements.txt", "requirements.txt", "./"]

RUN apt-get update \
 && xargs -d '\n' -a apt-requirements.txt apt-get install -y \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f requirements.txt apt-requirements.txt

COPY ["grad_cam", "./"]

CMD ["grad_cam.py", "--help"]
