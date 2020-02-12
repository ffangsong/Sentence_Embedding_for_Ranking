# Sentence-Embedding-for-Ranking
This project attempts to build a fast document retrieval system using sentence embedding. 

It allows to find the top K related docs in the repository for a given query.
The App is runing online at   . 


## Getting Started:
Clone this reposistory locally and create a virtual enviroment(conda examples below)

Create the environment file from the the provided environment.yaml file

    conda create python=3.6
    pip install -r requirement.txt
## Repository Structure


## How to install
Download a pre-trained Bert Model, the download example below is BERT-base, 

## Usage

## Run test
## Embedding Model Details
Two models were trained on a classification task to learn doc embeddings. 
* The first model leverages word embedding, Google's pretrained word2vec embedding was used and an LSTM layer was added to capture the long term dependency of words and capture richer semantic information.
* The second model start with a Bert layer initialized with the pre-trained weights and fine-tuned for the classification task.

In both models, a simple cosine similarity metric is used for classification, thus compile the model to learn a better embedding of the doc. 

Once the training is completed, the classfication layer is droped,  and the output of the last second layer is used as the embedding. 

## Indexing and Ranking

Annoy is used to create searchable index for ranking. 

