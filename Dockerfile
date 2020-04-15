FROM seunglab/chunkflow:pytorch
WORKDIR "/root/workspace"
COPY requirements.txt /root/workspace
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
RUN pip install  chunkflow==0.6.3
RUN pip install torch==1.3.1
RUN mkdir /nets && mkdir -p /root/workspace/pytorch-3dunet
RUN apt-get update && apt-get install -y wget parallel
COPY . /root/workspace/pytorch-3dunet
RUN conda develop /root/workspace/pytorch-3dunet
