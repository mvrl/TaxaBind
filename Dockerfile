FROM daskdev/dask

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update \
&& apt-get install -y build-essential \
&& apt-get install -y wget \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create a new conda environment and install all requirements 
RUN conda create -n ecobind python=3.10 && \
source /opt/conda/bin/activate ecobind && \
pip install -r requirements.txt

CMD ["/bin/bash"]