FROM ubuntu:22.04

# Perform the basic setup
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get upgrade -y
RUN apt-get install wget git -y

# Install Python (miniconda) and create a new environment
RUN mkdir -p /opt/miniconda3 &&\
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh &&\
	bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 &&\
	rm -rf /opt/miniconda3/miniconda.sh
RUN /opt/miniconda3/bin/conda upgrade -y conda && /opt/miniconda3/bin/conda create -y --name pyserv python=3.10
SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "pyserv", "/bin/bash", "-c"]

# Requires a NVIDIA GPU with Compute Capability >= 7.5, see https://developer.nvidia.com/cuda-gpus
RUN conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the required Python packages
RUN pip install fastapi uvicorn[standard] gunicorn python-multipart
RUN pip install transformers==4.37.2
RUN pip install ctranslate2==3.24.0 OpenNMT-py==2.3.0 wtpsplit

# Define the MT engine to use
ARG HF_MODEL_STR="facebook/nllb-200-3.3B"
ENV HF_MODEL_STR=$HF_MODEL_STR

WORKDIR /mt

# The code will fail if the $HF_MODEL_STR is not supported by CTranslate2, see https://opennmt.net/CTranslate2/guides/transformers.html
ENV TRANSFORMERS_CACHE="/mt/_cache"
RUN ct2-transformers-converter --model $HF_MODEL_STR --output_dir /mt/_model --quantization int8
# To save space, remove the model files after the Ctranslate2 conversion
RUN rm -rf /mt/_cache/

# If set to True, the app will validate the language codes passed in the request
# To do so, a supported_languages.txt file must be provided
# The file should contain a list of supported language codes, one per line
# The provided one works for the facebook/nllb-200* model family and was created based on the HF/Transformers documentation
ARG VALIDATE_LANGS="True"
ENV VALIDATE_LANGS=$VALIDATE_LANGS
# By using the wildarcd here, the build will not fail if the file is not present 
COPY supported_languages.tx* /mt 

# If enabled, a sentence splitter will be used to split the input text into sentences
ARG SPLIT_SENTENCES="True"
ENV SPLIT_SENTENCES=$SPLIT_SENTENCES

# Sets the log level for the app
ARG LOG_LEVEL="info"
ENV LOG_LEVEL=$LOG_LEVEL

# For safety, we limit the maximum file size that can be uploaded
# The defult value corresponds to 10MB (10*1024*1024)
ARG MAX_UPLOAD_FILE_SIZE="10485760"
ENV MAX_UPLOAD_FILE_SIZE=$MAX_UPLOAD_FILE_SIZE

COPY app.py /mt
COPY gunicorn_logging.conf /mt/logging.conf

CMD gunicorn app:app \
	--timeout 0  \
	--bind 0.0.0.0:8081 \
	--worker-class uvicorn.workers.UvicornWorker \
	--log-config /mt/logging.conf \
	--log-level $LOG_LEVEL
