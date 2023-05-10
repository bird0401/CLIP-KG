FROM python:3.10

# to process fast
ENV TERM xterm

# install python external modules
COPY requirements.txt /usr/src
WORKDIR /usr/src
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools && \
    pip install --no-cache-dir -r requirements.txt 
