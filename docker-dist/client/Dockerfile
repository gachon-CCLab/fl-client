# FROM nvidia/cuda:11.7.0-base-ubuntu22.04
FROM nvidia/cuda:11.7.0-base-ubuntu22.04
WORKDIR /code
ENV APP=client.py
ENV RUN_HOST=0.0.0.0
# RUN apt-get -y update \
#     && apt-get install -y software-properties-common \
#     && apt-get -y update \
#     && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install gcc
RUN apt-get -y install python3.8-dev
RUN apt-get -y install python3-pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8002
COPY . .
ENTRYPOINT [ "python3", "client.py" ]