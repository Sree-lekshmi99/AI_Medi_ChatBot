# Medical Chatbot - Medii 🏥🤖
![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red)
![Flask](https://img.shields.io/badge/Flask-2.0.1-lightgrey)
![NLP](https://img.shields.io/badge/NLP-NLTK3.6.5-green)

## 🌟 Features
- Symptom analysis using NLP
- Disease prediction with machine learning
- Severity assessment
- Medical recommendations
- Interactive web interface

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/yourusername/medii.git
cd medii
```

# Install dependencies
``` python
pip install -r requirements.txt
```

# Download NLTK data
```python
 -m nltk.downloader punkt
```
## 🚀 Usage
```python
# Start Flask application
from flask import Flask
app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)

```

## 🧠 NLP Model Architecture
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
```
## 🔍 Symptom Processing

```python

def bag_of_words(tokenized_sentence, all_words):
    stemmer = PorterStemmer()
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
```
## 🌐 Flask Endpoints
```python
@app.route('/symptom', methods=['POST'])
def predict_symptom():
    sentence = request.json['sentence']
    symptom, prob = get_symptom(sentence)
    if prob > .5:
        response = f"{(prob * 100):.2f}% confidence: {symptom}"
        user_symptoms.add(symptom)
    return jsonify(response)
```
📂 Project Structure
```
medii/
├── models/              # Trained models
├── data/               # Medical datasets
├── nnet.py             # Neural network class
├── nltk_utils.py       # NLP functions
├── app.py              # Flask application
└── templates/          # Web interface
```
##📚 Data Sources

Symptom-Disease mappings

Disease descriptions

Precaution recommendations

Symptom severity weights

## 🤖 Machine Learning Models
- **Neural Network (PyTorch)**

- **k-Nearest Neighbors**

- **Decision Tree**

- **Logistic Regression**

- **SVM**

- **Stacking Ensemble**

⚠️ Important Note FOR SEVERITY ANALYSIS
```python

if np.mean(severity) > 4 or np.max(severity) > 5:
    response += "<br>Consult a real doctor immediately!"
```
