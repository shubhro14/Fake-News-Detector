from flask import Flask, render_template, request
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__,template_folder='templates')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the model
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_article = request.form['article']
    new_article = new_article.lower()
    new_article = re.sub('[^A-Za-z0-9\s]', '', new_article)
    new_article = re.sub('\n', '', new_article)
    new_article = re.sub('\s+', ' ', new_article)
    sequences = tokenizer.texts_to_sequences([new_article])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)[0][0]
    if prediction >= 0.5:
        result = 'FAKE'
    else:
        result = 'REAL'
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
