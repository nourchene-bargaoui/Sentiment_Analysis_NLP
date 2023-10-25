# CMSC 516 Twitter/YouTube Sentiment Analysis

## Content

| Section | Description |
|-|-|
| [Project description](#project_description) | Detailed description of the project |
| [Installation](#installation_instructions) | How to install the package |
| [Overview](#overview) | Overview of the package |
| [Method](#method) | Method used for codes |
| [Doc](#doc) |  Detailed documentation |
| [Notebooks](#notebooks) | Introduction on the provided Jupyter Notebooks |
| [Data](#data) | Data used to train the models |
| [Results](#results) | Results of the models |

## Project Description
Our Project utilizes the BERT model and PyTorch to provide a sentiment analysis of our YouTube Comments/tweets dataset.

In our project, we analyze YouTube comments provided by Kaggle and we evaluate whether they are positive, negative or neutral with the pre-trained model BERT.
We compare the results obtained with YouTube Comments dataset and the results obtained from Twitter users replying to (or commenting on) YouTube Tweets, using Tweepy.

Our Notebook pulls the required data from the Github repository in order to analyze the comments. We used Pre-trained BERT model to score the sentiment (positive,negative,neutral) per comment and then averaged them per video. Once the model has completed training we run it against the tweets pulled from our Access_Twitter_API.ipynb in the file src/replies.csv.

## Overview

This package comprises the following classes that can be imported in Python :

  - [`BertModel`](https://huggingface.co/docs/transformers/model_doc/bert) - raw BERT Transformer model (**fully pre-trained**),
  - [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert) - BERT Transformer with a sequence classification head on top (BERT Transformer is **pre-trained**, the sequence classification head **is only initialized and has to be trained**),
  - [`BertTokenizer.from_pretrained`](https://huggingface.co/docs/transformers/model_doc/bert) - BERT Transformer with a token classification.
- The **Transformer** PyTorch models (`torch.nn.Module`) 
- Optimizer for **BERT**  `Adam` - Bert version of Adam algorithm with warmup and linear decay of the learning rate.

## Installation and Usage Instructions
This repo was tested on Python 2.7 and 3.5+ 
This project was built and tested in Google Colab. It can be ran the same way. To run this project yourself:
1. Follow this link to the Google Colab - https://colab.research.google.com/
2. Go to the Colab task bar and click File > Open Notebook
3. Click on GitHub
4. Search for https://github.com/NathanH-VCU/Sentiment-Analysis-Nourchene-Nathan/blob/main/BERT_sentiment_analysis_using_Youtube_Comments.ipynb
5. Click on BERT_sentiment_analysis_using_Youtube_Comments.ipynb
6. Click Runtime
7. Click Run All
8. Authorize the notebook
9. Wait until the process finishes, our tests took about 40 minuets with our sample data.

## Method
1. For the code for Sentiment Analysis using BERT titled : 
BERT_sentiment_analysis_using_Youtube_Comments.ipynb
-	Feature representation : Bert contextual embedding
-	Python Algorithm : Gradient Descent machine Learning algorithm
-	Dataset : YouTube comments from Kaggle

## Data
For our data we are utilizing Kaggle's YouTube Statistics dataset, specifically the comments.csv file. This data set contains a list of Video ID's, comments, likes, and the Sentiment of each comment. For our project we are only utilizing the comments and their sentiment for training and developing our model.
The size of our data set is 18,409 comments with 12.7% negative sentiment, 25% neutral sentiment, and 62% positive sentiment. 
For our twitter data we are using tweepy to pull from the twitter api all replies to Youtube tweets. This data includes the origional tweet from YouTube, the Reply that is 6 words or longer and the user who replied. The only information we need from this is the reply.

## Doc

Here is a detailed documentation of the papers and links we used for this project:

| Reference | Link |
|-|-|
| Feature Extraction for NLP | https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8999624&tag=1 |
| NLP based sentiment analysis on Twitter | https://ieeexplore.ieee.org/document/7219856 |
| Deep Learning Model-Based Approach for Twitter Sentiment Classification | https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9373371 |

## Notebooks

We include two Jupyter Notebooks for the code

- The first NoteBook ([BERT_sentiment_analysis_using_Youtube_Comments.ipynb](./BERT_sentiment_analysis_using_Youtube_Comments.ipynb)) is used to test the sentiment analysis on YouTube comments Kaggle data.

- The second NoteBook ([Sentiment_Analysis.ipynb](./Sentiment_Analysis.ipynb)) used to test the sentiment analysis on YouTube tweets data.

## Results
1. For Youtube comments 
The first code "BERT_sentiment_analysis_using_Youtube_Comments.ipynb" we got these results.

Our BERT model has giving us 70% accuracy on the test data.

![image](https://user-images.githubusercontent.com/83011466/196296921-76b9cbfa-6e26-47b5-b8f0-f9efe0b4b8aa.png)

2. For Youtube tweets and predicting sentiment
The code Sentiment Analysis.ipynb gave us a low accuracy of 22%.
Additionally each twitter reply sentiment prediction produced a negative result.

## Discussion
1. For the first code about Sentiment Analysis using YouTube comments.
This project consists of creating a machine learning algorithm for classifying text according to their Sentiment "postive", "negative" and "Neutral".
We have demonstrated how BERT is the new revolution in Natural Language processing especially in Sentiment Analysis.
Additionally, a Gradient-Descent algorithm was developed for this classification task.
We can see that the model gave us good results of 70% accuracy. The F1_Score, Precision and Recall functions also gave good results.

2. While Sentiment Analysis.ipynb is definitly not our flagship notebook we wanted to provide some results that would show the difference between the two platforms, YouTube and Twitter. While the accuracy is very low the results could still show that those who comment on Twitter regarding Youtube is often with a negative sentiment (directed towards YouTube) than those who generally coment on YouTube videos (sentiment directed toward YouTube Content Providers).

## Future Work
This work can be used for a "Market Making" strategy. Indeed, it can be further developed to extract youtube comments and youtube replies about a specific brand or company to determine the feedback of customers and users. We want to use more complex algorithms on a more enriched dataset in the future.
