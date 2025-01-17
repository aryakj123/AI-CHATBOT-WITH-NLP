**Company:** CODTECH IT SOLUTIONS

**ID:** CT08EQI

**Domain:** Python Programming

**Duration:** December 20,2024-January 20,2025

**Mentor:** SRAVANI GOUNI


### Description of Task ###
# Python Chatbot with NLTK/SpaCy: A Simple Intent Recognition System

This repository contains a Python implementation of a basic chatbot that utilizes Natural Language Processing (NLP) to understand user input and provide relevant responses. The goal of this project is to demonstrate the fundamental concepts of intent recognition using a simple rule-based approach with pre-defined intents and responses. This is intended for educational purposes and as a stepping stone for those interested in learning about building more advanced conversational AI systems.

## Core Concepts

This chatbot works by following these key steps:

1.  **Data Preparation:** The chatbot relies on a pre-defined dictionary (`intents`) where each "intent" (e.g., "greeting", "goodbye") is associated with example "patterns" (how a user might express that intent) and corresponding "responses." This forms the training data for our intent recognition.

2.  **NLP Preprocessing:** The input text, whether it's a user message or a pattern from the training data, goes through several NLP preprocessing steps:
    *   **Tokenization:** The text is broken down into individual words or "tokens."
    *   **Lowercasing:** All text is converted to lowercase to ensure consistent matching.
    *   **Lemmatization:** Each word is reduced to its root form (lemma) to group similar words (e.g., "running," "ran," and "runs" all become "run").
    *   **Stop Word Removal:** Common words like "a," "the," "is," etc. are removed since they don't typically carry significant meaning.
    *   **Punctuation Removal:** Punctuation marks are also removed.

    These steps are done using the SpaCy library.
3.  **Intent Recognition:** Once a user message has been preprocessed, the chatbot does the following:

    *   **Vectorization:** The preprocessed input, along with all the preprocessed training patterns, is converted into a numerical representation using TF-IDF (Term Frequency-Inverse Document Frequency). This converts text into vectors where each dimension represents a word. The training set is fit and transformed, and then the new input is transformed.
    *   **Similarity Comparison:** The bot compares the input vector to all the pattern vectors from the training data, using cosine similarity. Cosine similarity is used to find the closest matching intent based on the angle between the vectors.
    *   **Best Match:** The intent with the highest similarity to the user's input is considered the best match, indicating the user's intention.

4.  **Response Generation:** Once an intent is recognized, the chatbot selects a response from the set of pre-defined responses for that specific intent. A random response is chosen so that the chatbot does not provide the same response every time.

5.  **Chatbot Logic:**
   * The main function sets up the TF-IDF vectorizer and then enters a loop, where it waits for a user input, or ends the chatbot loop if the user input is 'exit'. After recieving input, the chatbot predictes the intent, generates a response, and prints out the response.

## Code Structure

The repository contains a single Python script (`your_script_name.py`) that defines all the chatbot logic:

*   **Data Storage:** The `intents` dictionary houses the training data with intents, patterns, and responses.
*   **Preprocessing Function:** The `preprocess_text()` function uses SpaCy to clean the text.
*   **Intent Prediction Function:** The `predict_intent()` function does vectorization, similarity calculation, and prediction using TF-IDF and cosine similarity.
*   **Response Selection Function:** The `get_response()` function selects a relevant response from the data.
*   **Chatbot Loop:** The `chatbot()` function handles the main interaction loop.

## Limitations

This chatbot is very basic and has the following limitations:

*   **Limited Vocabulary:** The chatbot can only understand intents that are explicitly defined in the `intents` dictionary. It does not have general knowledge.
*   **No Conversational Context:** It doesn't remember previous interactions, so it can't handle follow-up questions or context-dependent conversations.
*   **Simple Matching:** It uses simple text matching and cannot handle more nuanced or complex requests.
*   **Dependency on Exact Patterns:** Similar sentences might not be understood if they don't closely match predefined patterns.
*   **No Real-Time Information:** This chatbot doesn't fetch external information from APIs for weather, time, or anything else.

## Getting Started

1.  **Installation:**
    ```bash
    pip install nltk spacy scikit-learn
    python -m spacy download en_core_web_sm
    ```

2.  **Run the Chatbot:**
    ```bash
    python your_script_name.py
    ```

## Next Steps

This project provides a basis for a more complex system. Future improvements could include:

*   Expanding the training dataset.
*   Using word embeddings (Word2Vec, GloVe, BERT, etc.) for better semantic understanding.
*   Implementing more sophisticated machine learning models (SVM, neural networks, etc.) for intent classification.
*   Adding context management to handle more complex conversations.
*   Integrating external APIs for expanded functionality.
