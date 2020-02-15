# Sentence-Embedding-for-Doccument-Ranking
This repository is the implelentation of building a document ranking and retrieval system using the sentence embedding tailored for your sepecial corpus. 

It allows to you to train the sentence embedding system on your own unique corpus, create your own indexed document reposistory and buid a ranking system to  output the top K similar/relavant docs in the repository for a given input query.


## Demo:



## Getting Started:
Clone this reposistory locally and create a virtual enviroment(conda examples below)
```
git clone https://github.com/ffangsong/Sentence_Embedding_for_Ranking.git
cd Sentence_Embedding_for_Ranking
```

Create the environment file from the the provided requirement.txt file

    conda create python=3.6
    pip install -r requirement.txt
    
Make sure you have dataset in the ```data``` folder(you can specify the path in the bash script later)    
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

