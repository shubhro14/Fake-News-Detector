# Fake News Detection

This repository contains the code and dataset for a fake news detection model. The model is trained to classify news articles as either real or fake based on their content.

## Folder Structure

- `Dataset`: This folder contains the training dataset used to train the model. It includes labeled news articles for both real and fake news.

- `model_training.py`: This is the trained model which is being trained using LSTM.

- `Fake_news`: This folder contains the necessary files for implementing the fake news detection model.

  - `app.py`: This file is the Flask application that provides a user interface for interacting with the model. It allows users to input a news article and receive a prediction of its authenticity.

  - `tokeniser.pkl`: This file contains the serialized tokenizer object used to preprocess the text data before feeding it into the model.

  - `model.h5`: This file contains the trained model weights for the fake news detection model.

## Usage

1. Clone the repository to your local machine using `git clone https://github.com/your-username/Fake-News-Detection.git`.

2. Navigate to the `Fake_news` folder and ensure that you have the necessary dependencies installed.

3. Run the Flask application by executing the `app.py` file. This will start the server locally.

4. Access the application by opening your web browser and visiting `http://localhost:5000`. You will see a form where you can input a news article.

5. Submit the form, and the application will provide a prediction of whether the news article is real or fake.

## Video Demo

Watch the demo of the Fake News Detection model in action:

[Fake News Detection Demo](https://youtu.be/R7UVl4vR7k0)

## Note

The code in the repository is provided as a demonstration of the fake news detection model. 
