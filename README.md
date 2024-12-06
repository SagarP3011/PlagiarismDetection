# PlagiarismDetection
Plagiarism Detection System
Overview
The Plagiarism Detection System is a web-based tool designed to identify similarities between text documents, code files, and online content. It leverages advanced natural language processing (NLP) techniques, similarity algorithms, and web scraping to analyze input text or files and compare them with existing sources. The system generates a detailed similarity report, highlighting the matched content and potential source URLs.
Features
1.Multiple Input Options:
      a. Enter text directly.
      b. Upload files in formats like .txt, .pdf, .docx.
      c. Compare two uploaded files for similarity.
2.Similarity Detection:
      a. Text similarity analysis using cosine similarity.
      b. Sentence-level comparison for detailed matching.
3.Web Integration: Extract potential source URLs for suspected plagiarized content using web scraping or APIs.
4.Dynamic Visualization: Visualize similarity results using graphs and charts.
5.Future Scope: Extend functionality to detect plagiarism in programming code.

System Architecture
The system consists of:
1. Preprocessing Module: Tokenization, removal of stopwords, and lemmatization.
2. Similarity Computation: Uses cosine similarity to calculate document similarity.
3. Web Scraping: Retrieves possible source URLs for plagiarized content.

Technologies Used
1. Frontend: Streamlit (Web App Framework)
2. Backend: Python
3. Libraries:
    Natural Language Toolkit (NLTK)
    Scikit-learn (Cosine Similarity and TF-IDF)
    BeautifulSoup (Web Scraping)
    Plotly (Visualization)
4. API: Google Custom Search API
