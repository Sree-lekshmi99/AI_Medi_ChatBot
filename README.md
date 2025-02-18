# Meddy - Medical Chatbot 🤖

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-yellow)

## 📌 Overview

Meddy is a medical chatbot designed to provide preliminary disease predictions based on user-provided symptoms. Meddy leverages Natural Language Processing (NLP) techniques and machine learning models to analyze symptoms and suggest possible diseases along with precautions and descriptions.

## 🎯 Project Goals

-   Develop an NLP model to understand and extract symptoms from user input.
-   Create a predictive model that suggests possible diseases based on symptoms.
-   Provide users with relevant disease descriptions and precautions.
-   Build a user-friendly web interface using Flask for interaction with the chatbot.

## ⚙️ Technologies Used

-   **Python**: Primary programming language.
-   **PyTorch**: Deep learning framework for NLP model.
-   **NLTK**: Natural Language Toolkit for text processing.
-   **Scikit-learn**: Machine learning library for the prediction model.
-   **Flask**: Web framework for creating the chatbot interface.
-   **Pandas**: Data manipulation and analysis.
-   **NumPy**: Numerical computing.

## 📂 Data Sources

-   `intents_short.json`: Contains intents and patterns for training the NLP model.
-   `dataset.csv`: Dataset of diseases and symptoms.
-   `symptom_Description.csv`: Contains descriptions of diseases.
-   `symptom_precaution.csv`: Provides precautions for different diseases.
-   `Symptom-severity.csv`: Contains the severity weights of symptoms.
-   `list_of_symptoms.pickle`: Pickled list of symptoms.

## 🧱 Project Structure

├── data/
│ ├── dataset.csv
│ ├── intents_short.json
│ ├── list_of_symptoms.pickle
│ ├── Symptom-severity.csv
│ ├── symptom_Description.csv
│ └── symptom_precaution.csv
├── models/
│ ├── data.pth
│ └── fitted_model.pickle2
├── static/
│ └── assets/
│ └── files/
│ └── ds_symptoms.txt
├── nnet.py
├── nltk_utils.py
├── app.py
├── README.md
└── ...


-   `data/`: Contains datasets and JSON files used for training and prediction.
-   `models/`: Stores the trained NLP model (`data.pth`) and the disease prediction model (`fitted_model.pickle2`).
-   `static/`: Includes static files for the Flask web application.
-   `nnet.py`: Defines the neural network architecture for the NLP model.
-   `nltk_utils.py`: Contains utility functions for NLP tasks.
-   `app.py`: The main Flask application file.
-   `README.md`: Project documentation.

## 🛠️ Setup and Installation

1.  **Clone the repository:**

    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment:**

    ```
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   On Windows:

        ```
        venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**

    ```
    pip install -r requirements.txt
    ```

    Create `requirements.txt` file:

    ```
    torch
    nltk
    scikit-learn
    Flask
    pandas
    numpy
    ```

5.  **Download NLTK resources:**

    ```
    import nltk
    nltk.download('punkt')
    ```

## 🚀 Running the Chatbot

1.  **Navigate to the project directory:**

    ```
    cd <repository_directory>
    ```

2.  **Run the Flask application:**

    ```
    python app.py
    ```

3.  **Access the chatbot:**

    Open your web browser and go to `http://127.0.0.1:5000` to interact with Meddy.

## 🧠 NLP Model Training

### Data Preparation

1.  **Load the intents data:**

    The `intents_short.json` file contains the patterns and tags used for training the NLP model.
2.  **Tokenize and stem the words:**

    Use NLTK to tokenize the sentences and apply stemming to reduce words to their root form.
3.  **Create a bag of words:**

    Convert the tokenized sentences into a bag of words representation.
4.  **Prepare the training data:**

    Create the training data by converting the patterns into bag of words and assigning labels to the tags.

### Neural Network Model

1.  **Define the dataset:**

    Create a custom dataset class for handling the training data.
2.  **Define the neural network architecture:**

    Create a neural network with linear layers and ReLU activation functions.
3.  **Train the model:**

    Instantiate the model, define the loss function and optimizer, and train the model on the training data.
4.  **Save the trained model:**

    Save the model's state dictionary and other necessary information for later use.

## 🩺 Disease Prediction Model

1.  **Load and preprocess the disease data:**

    Load the `dataset.csv` and preprocess the symptoms.
2.  **Train a prediction model:**

    Use the dataset to train a model that predicts diseases based on symptoms.

## 💬 Chatbot Logic

1.  **Load the trained models and data:**

    Load the NLP model and the disease prediction model.
2.  **Implement the chatbot logic:**

    Use the NLP model to extract symptoms from user input and the disease prediction model to suggest possible diseases.

## 📈 Results

The final loss after training the NLP model is approximately 0.1783, indicating good convergence.

## ✍️ Conclusion

Meddy is a functional medical chatbot that can provide preliminary disease predictions based on user-provided symptoms. The project demonstrates the application of NLP and machine learning techniques in the healthcare domain. Further improvements can be made by expanding the dataset, refining the NLP model, and incorporating more advanced prediction algorithms.
