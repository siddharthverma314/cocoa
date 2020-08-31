FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	locales \
	cmake \
	git \
	curl \
	vim \
	unzip \
	ca-certificates \
	libjpeg-dev \
	libpng-dev \
	libfreetype6-dev \
	libxft-dev &&\
	rm -rf /var/lib/apt/lists/*

# install conda to /opt/conda
RUN curl -o ~/miniconda.sh -O "https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh" && \
	chmod +x ~/miniconda.sh && \
	~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

# create base environment
RUN /opt/conda/bin/conda install -y python=2.7 numpy pyyaml scipy ipython mkl mkl-include cython typing && \
	/opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
	/opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


# install dependencies
RUN conda install -c pytorch pytorch=0.4.1 cuda90
RUN conda install numpy=1.13.3=py27hdbf6ddf_4
RUN pip install \
	pandas==0.20.3 \
	matplotlib==2.0.2 \
	flask==0.12.2 \
	flask-socketio==2.8.5 \
	nltk==3.2.4 \
	ujson==1.35 \
	decorator==4.1.2 \
	future==0.16.0 \
	nose==1.3.7 \
	scikit-learn==0.19.0 \
	sklearn==0.0 \
	torchtext==0.2.1 \
	visdom==0.1.6.1
RUN python -m nltk.downloader punkt && \
	python -m nltk.downloader stopwords

# install cocoa
COPY ./ cocoa/
RUN cd cocoa && python setup.py develop


# make workdir
WORKDIR /cocoa/
ENV PYTHONPATH /cocoa/

# download craigslist data
RUN cd craigslistbargain/data/ && \
	curl -o train.json -O https://worksheets.codalab.org/rest/bundles/0xda2bae7241044dbaa4e8ebb02c280d8f/contents/blob/ && \
	curl -o dev.json -O https://worksheets.codalab.org/rest/bundles/0xb0fe71ca124e43f6a783324734918d2c/contents/blob/ && \
	curl -o test.json -O https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/

# compute artifacts
RUN cd craigslistbargain && python core/price_tracker.py --train-examples-path ./data/train.json --output price_tracker.pkl
RUN cd craigslistbargain && \
	python parse_dialogue.py \
	--transcripts data/train.json \
	--price-tracker price_tracker.pkl \
	--max-examples -1 \
	--templates-output ./data/train_templates.pkl \
	--model-output ./data/train_model.pkl \
	--transcripts-output ./data/train_parsed.json
RUN cd craigslistbargain && \
	python parse_dialogue.py \
	--transcripts data/dev.json \
	--price-tracker price_tracker.pkl \
	--max-examples -1 \
	--templates-output ./data/dev_templates.pkl \
	--model-output ./data/dev_model.pkl \
	--transcripts-output ./data/dev_parsed.json
RUN cd craigslistbargain && \
	python parse_dialogue.py \
	--transcripts data/test.json \
	--price-tracker price_tracker.pkl \
	--max-examples -1 \
	--templates-output ./data/test_templates.pkl \
	--model-output ./data/test_model.pkl \
	--transcripts-output ./data/test_parsed.json

# add vocab.pkl and config.json
RUN mkdir -p craigslistbargain/mappings/lf2lf && \
	cd craigslistbargain/mappings/lf2lf && \
	curl -o vocab.pkl -O https://worksheets.codalab.org/rest/bundles/0xab2055ab75de4c9c825a804795ddb120/contents/blob/mappings/lf2lf/vocab.pkl
RUN mkdir -p craigslistbargain/checkpoint/lf2lf && \
	cd craigslistbargain/checkpoint/lf2lf && \
	curl -o config.json -O https://worksheets.codalab.org/rest/bundles/0xab2055ab75de4c9c825a804795ddb120/contents/blob/checkpoint/lf2lf/config.json

# download base model
RUN cd craigslistbargain/checkpoint/lf2lf && \
	curl -o model_best.pt -O https://worksheets.codalab.org/rest/bundles/0xab2055ab75de4c9c825a804795ddb120/contents/blob/checkpoint/lf2lf/model_best.pt

# download all finetuned models
RUN mkdir craigslistbargain/checkpoint/lf2lf-margin && \
	cd craigslistbargain/checkpoint/lf2lf-margin && \
	curl -O https://worksheets.codalab.org/rest/bundles/0xd658de343912461598ce53dc9354dc60/contents/blob/checkpoint/lf2lf-margin/model_best.pt
RUN mkdir craigslistbargain/checkpoint/lf2lf-fair && \
	cd craigslistbargain/checkpoint/lf2lf-fair && \
	curl -O https://worksheets.codalab.org/rest/bundles/0x5120cbb5102e41058c66a80fc34107d1/contents/blob/checkpoint/lf2lf-margin/model_best.pt
RUN mkdir craigslistbargain/checkpoint/lf2lf-length && \
	cd craigslistbargain/checkpoint/lf2lf-length && \
	curl -O https://worksheets.codalab.org/rest/bundles/0x5120cbb5102e41058c66a80fc34107d1/contents/blob/checkpoint/lf2lf-margin/model_best.pt
