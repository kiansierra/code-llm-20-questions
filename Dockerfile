FROM gcr.io/kaggle-gpu-images/python:latest as base

RUN apt install pigz pv 

WORKDIR /build

RUN pip install --upgrade pip

COPY requirements.txt .
COPY requirements_build.txt .

RUN pip install -r requirements_build.txt --target /libs

RUN pip install -r requirements.txt 

# RUN pip install flash-attn==2.5.1.post1 --no-build-isolation --target /libs

COPY . .

RUN mkdir -p subs

RUN pip install . --target /libs

RUN pip install -e . 
