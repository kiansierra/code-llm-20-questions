FROM gcr.io/kaggle-gpu-images/python:latest as base

RUN apt install pigz pv 

WORKDIR /build

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY requirements_build.txt .

RUN pip install -r requirements_build.txt --target /libs

COPY . .

RUN mkdir -p subs

RUN pip install . --target /libs

RUN pip install -e . 
