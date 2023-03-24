# ZAF202303_NLP_NewsClassifier

## Project Goal

We are interested in building a high quality text classifier that categorizes new stories into 2 categories.

## Overview
We are building a web application with API access to a machine learning model without giving source of code or any details how it works. We are interested in building a high quality text classifier that categorizes new stories into 2 categories - information and entertainment. We want the classifier to stick with predicting the better among these two categories (this classifier will not predict a percentage score for these two categories).

## Methodology

There are 10000 different news stories and additional news stories.
1. Run the classifier on 1 mln random stories out of 1000 news sources. Get 10k stories where the classifier output is the closest to the decision boundary and get these examples labeled.
2. Get 10k random labeled stories from the 10k random labeled stories from 1000 news we care about.
3. Pick a random sample of 1 mln stories from 1000 news sources and have them labeled. Pick the subset of 10k stories where the model’s output is both wrong and farthest away from the decision boundary.
4. Using different methods of classifying a bag of news articles from 1000 news sources, measure the accuracy of the model with an original train dataset.

## AI Application
- NLP

## Business Segments
- Media & Publishing




### Use Case can be found [here](https://docs.google.com/document/d/1zDueu4PD7Nwj_7dxiXWyluaCejUx2fpI2ebWugNL39k/edit?usp=sharing).


https://user-images.githubusercontent.com/95337849/227604177-368580bf-e63d-48c2-a5a3-b37d33c453cd.mp4

