FROM ubuntu:20.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3.8 python3-pip

RUN pip3 install --trusted-host pypi.python.org setuptools

WORKDIR /app
COPY . /app

RUN pip3 install -e . 

ENTRYPOINT ["python3", "-m", "recsys.api"]

