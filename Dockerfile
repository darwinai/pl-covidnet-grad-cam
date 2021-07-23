FROM fnndsc/ubuntu-python3:18.04
MAINTAINER fnndsc "dev@babymri.org"

ENV APPROOT="/usr/src/grad_cam"
ENV DEBIAN_FRONTEND=noninteractive
COPY ["requirements.txt", "${APPROOT}"]

WORKDIR $APPROOT

RUN apt-get update \
  && apt-get install -y libsm6 libxext6 libxrender-dev python3-tk\
    && pip install --upgrade pip \
      && pip install -r requirements.txt

COPY ["grad_cam", "${APPROOT}"]

CMD ["grad_cam.py", "--help"]
