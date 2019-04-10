#!/usr/bin/env bash

# Create and setup the virtual environment with required dependancies
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Create directory for dataset storage
DIR_NAME=$(PWD)
mkdir $DIR_NAME/data
mkdir $DIR_NAME/data/embed
mkdir $DIR_NAME/data/wikiqa

# Download word embeddings
URL_GLOVE='http://nlp.stanford.edu/data'
FILENAME='glove.6B.zip'
DIR_EMBED=$DIR_NAME/data/embed
echo $'Downloading word embeddings ...\n'
python setup_helper.py --url $URL_GLOVE --filename $FILENAME --dir $DIR_EMBED
