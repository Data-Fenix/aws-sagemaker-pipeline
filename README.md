
# Orchestrating Cutomer Churn Prediction ML Workflow with AWS SageMaker Pipelines


## Table of content

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)
## Demo
![full](https://github.com/Data-Fenix/aws-sagemaker-pipeline/blob/main/demo/full.gif)

## Overview

A ML workflow is a sequence of ML related taks that runs in a sequential manner. This workflow help to automate the overal MLOps workflow, from data ingestion, prepreration, training, optimizing and deployment. So in this notebook will demostrate a simple end to end ML pipepline workflow deployment using AWS Sagemaker.

ML use case of this project is, identify the churn cutomers in telco compnay and identifying those customers' saves millions of money to the company.

In here used only the XG boost model for the model training because our ultimate goal of this project is to demostrate how to deploy ML model in AWS Sagemaker.

Special Note:

We bring our own scripts into deployment process and this method called bring your own code concept. Therefore must do the containerization and in the inference job we used FlaskApp. 
## Dataset

This is data gathered from 7043 telco customers and dataset has 21 features (columns). Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The “Churn” column is our target variable and it has two outcomes. Therefore this is a binary classification problem and using below link,you can easily download the dataset.
https://www.kaggle.com/blastchar/telco-customer-churn
## Motivation

When I was searching about AWS Sagemaker, I struggle a lot as lack of references in this feild. It has some references, but there are missing few things. Therefore I need to fullfil that gap. So now I have some experience in this feild and as a MLOps team memeber, I migrated a lot of projects into cloud. I saved data scientists' valuable time by automating and scheduling their ML projects. So now I need to share that experience and knowledge with you and this is my first step of that journey.
## Technical Aspects

This project has major two parts:
1. Training a ML model using XGBoost. (from scratch)
2. Deploying end to end ML workflow using AWS Sagemaker and divided the whole script into below componenents.
    - Data preprocessing
    - Training
    - Store the models in Model Registry
    - Inferencing (Batch transformation done by using FlaskApp)
## Installation

#### Requirements

1. An AWS account
2. Python 3.5+
3. Docker (optional)


Only thing you need to satisfy in this list is you must have an AWS account. If you don't have an account you can create it free, using below link:
https://aws.amazon.com/free/
    
## Run
## Deployment on AWS Sagemaker


## Directory Tree

```
├── inferencing 
│   ├── model
|       ├── nginx.conf
|       ├── predictor.py
|       ├── serve.py
|       └── wsgi.py
│   ├── Dockerfile
├── preprocessing
│   ├── Dockerfile
|   └── preprocessing.py
├── training
│   ├── model
|       └──train.py
|   ├── Dockerfile
├── build_docker.ipynb
├── sagemaker_pipeline.ipynb
├── aws_helper.py
├── CONTRIBUTING.md
├── LICENSE
├── setup.py
├── tox.ini
└──WA_Fn-UseC_-Telco-Customer-Churn.csv
```
## Technologies Used
## Feature Requests
[<img target="_blank" src="https://venturebeat.com/wp-content/uploads/2021/02/SageMaker.jpg?fit=1292%2C664&strip=all" width=200>](https://venturebeat.com/wp-content/uploads/2021/02/SageMaker.jpg?fit=1292%2C664&strip=all) [<img target="_blank" src="https://www.cloudsavvyit.com/p/uploads/2019/06/55634f08.png?width=1198&trim=1,1&bg-color=000&pad=1,1" width = 200>](https://www.cloudsavvyit.com/p/uploads/2019/06/55634f08.png?width=1198&trim=1,1&bg-color=000&pad=1,1) [<img target="_blank" src="https://jfrog.com/connect/images/6053d4dc2f6c53160a53d407_linux-container-updates-iot.png" width = 200>](https://jfrog.com/connect/images/6053d4dc2f6c53160a53d407_linux-container-updates-iot.png) [<img target="_blank" src="https://logos-world.net/wp-content/uploads/2021/02/Docker-Logo-2015-2017.png" width = 200>](https://logos-world.net/wp-content/uploads/2021/02/Docker-Logo-2015-2017.png) [<img target="_blank" src="https://miro.medium.com/max/438/1*0G5zu7CnXdMT9pGbYUTQLQ.png" width = 200>](https://miro.medium.com/max/438/1*0G5zu7CnXdMT9pGbYUTQLQ.png) [<img target="_blank" src="https://logos-world.net/wp-content/uploads/2021/10/Python-Symbol.png" width = 200>](https://logos-world.net/wp-content/uploads/2021/10/Python-Symbol.png)
## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


