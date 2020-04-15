FROM seunglab/chunkflow:pytorch
RUN pip install --upgrade pip
RUN pip install torch scikit-learn tensorboardx h5py scipy scikit-image scikit-learn pyyaml pytest
RUN pip install hdbscan chunkflow==0.6.3
RUN mkdir /nets && mkdir -p /root/workspace/pytorch-3dunet
WORKDIR "/root/workspace"
COPY . /root/workspace/pytorch-3dunet
