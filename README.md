Text Summarization Project — Extractive & Abstractive Methods
This repository contains my graduation project, where I explored and implemented both extractive and abstractive text summarization techniques. The goal of the project was to build systems capable of generating concise and meaningful summaries from longer text documents.

📚 Project Overview
Text summarization is a core task in Natural Language Processing (NLP) with a wide range of applications like news summarization, content previews, and document analysis. In this project, I compared and implemented two main approaches:

Extractive Summarization: Selecting the most important sentences or phrases directly from the source text.

Abstractive Summarization: Generating new sentences that capture the essence of the original content using natural language generation techniques.

🧠 Objectives
Understand and implement NLP preprocessing techniques

Build extractive summarization models using rule-based and ML approaches

Implement abstractive summarization using transformer-based models

Compare the performance of both techniques using evaluation metrics

🔨 Tools & Technologies
Python

NLTK, spaCy

Scikit-learn

Hugging Face Transformers (T5, BART)

TensorFlow / PyTorch

Jupyter Notebooks

🔄 Workflow
1. Data Collection
Used datasets such as the CNN/DailyMail or other open-source text corpora for training and evaluation.

2. Preprocessing
Tokenization, stop word removal

Lemmatization and lowercasing

Sentence segmentation for extractive summarization

3. Extractive Summarization
Applied techniques like:

Frequency-based scoring

TF-IDF vectorization

Cosine similarity

Selected top-ranked sentences to form the summary

4. Abstractive Summarization
Fine-tuned or used pretrained models like:

T5 (Text-to-Text Transfer Transformer)

BART (Bidirectional and Auto-Regressive Transformer)

Generated summaries based on maximum input tokens and adjusted decoding strategies (e.g., beam search)

5. Evaluation
Compared summaries using:

ROUGE-1, ROUGE-2, and ROUGE-L scores

Human readability and relevance checks

📊 Results
Extractive methods provided fast and accurate summaries for well-structured texts.

Abstractive methods produced more natural and readable summaries, though required more computational resources.

ROUGE scores showed improvements when using fine-tuned transformer models.

📁 Repository Structure
extractive_summarization.ipynb — All steps for extractive methods

abstractive_summarization.ipynb — Hugging Face-based summarization

evaluation.ipynb — Evaluation metrics and comparison

README.md — This file

🧠 Skills Developed
Deep understanding of NLP pipelines

Experience with both classical and modern summarization techniques

Hands-on practice with Transformer models and Hugging Face

Critical thinking on model evaluation and optimizatio
