FROM python:2.7-slim

RUN apt-get update && apt-get install -y \
	build-essential \
	apt-utils \
	git \
	libglib2.0 \	
	tk \
	screen \
	vim && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r /requirements.txt

# set up app
COPY files /files

WORKDIR /files
