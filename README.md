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


## Train Embedding Model Details
Two models can be trained to learn doc embeddings. 
* The first model leverages pretrained word embedding, please download Google's pretrained model [here](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz) and put in the ```docs\Pretrained``` fodler. The word embeddings were them fed into a LSTM layer to capture the long term dependency of the words and thus richer semantic informations. 
* The second model start with a Bert layer initialized with the pre-trained weights , followed by a pool layer and  and a drop out layey. During training, the Bert model was fine-tuned for the special input corpus. Your can download pretrained Bert model [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and put in the ```docs\Pretrained```

In both models, a simple cosine similarity metric is used for classification, thus compile the model to learn a better docuemnt  embedding. 

Once the training is completed, the classfication layer is droped,  and the output of the last second layer is used as the embedding. 

## Indexing and Ranking

To generate the indexed docuemnt repositoty, please run:
```
python .\src\indexing.py
```
A indexed will generated and saved at ``` .\Docs ```


## Run test

To run test on the application, please run:
```
python .\src\ranking.py
```




## More information about 
Google's pretrained word2vec model includes word vectors for a vocabulary of 3 million words and phrases that they trained on roughly 100 billion words from a Google News dataset. The vector length is 300 features.
