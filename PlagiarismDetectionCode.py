# Importing Libraries
import streamlit as st  # For WebApp creation (UI)
import pandas as pd  # For data manipulation
import nltk  # For text processing
from nltk import tokenize
from nltk.corpus import stopwords
from string import punctuation
from bs4 import BeautifulSoup  # For web scraping
import requests  # For web scraping
from sklearn.feature_extraction.text import CountVectorizer  # Vectorize text
from sklearn.metrics.pairwise import cosine_similarity  # Compute similarity
import io  # I/O file handling
import docx2txt  # Document handling (.docx format files)
from PyPDF2 import PdfReader  # PDF handling
import plotly.express as px  # Plotting graphs
import time
import random
import os
import re

# Preload stopwords globally
STOP_WORDS = set(stopwords.words('english'))

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    # Split into words and filter out stopwords
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

# Breaking text into sentences
def get_sentences(text):
    return tokenize.sent_tokenize(text)

# WEB SCRAPING 
def get_url(sentence):
    """
    Fetch URLs related to a given sentence using web scraping and Google Custom Search API.
    Returns a list of URLs or an empty list if no results are found.
    """
    headers_list = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:40.0) Gecko/20100101 Firefox/40.0'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
    ]

    def scrape_google_search(query):
        try:
            base_scrape_url = 'https://www.google.com/search?q='
            scrape_url = base_scrape_url + query.replace(' ', '+')
            headers = random.choice(headers_list)
            response = requests.get(scrape_url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            divs = soup.find_all('div', class_='yuRUbf')
            return [div.find('a')['href'] for div in divs if div.find('a')]
        except Exception as e:
            print(f"Scraping Error: {e}")
            return []

    def google_custom_search(query):
        try:
            base_url = 'https://www.googleapis.com/customsearch/v1'
            api_key = "YOUR_API_KEY"  
            cse_id = "YOUR_CSE_ID"  

            params = {"q": query, "key": api_key, "cx": cse_id}
            response = requests.get(base_url, params=params, timeout=5)
            response.raise_for_status()
            results = response.json().get("items", [])
            return [item["link"] for item in results]
        except Exception as e:
            print(f"API Error: {e}")
            return []

    # Combine both approaches
    query = sentence.strip().replace(' ', '+')
    urls = scrape_google_search(query)
    if not urls:  # Fallback to API if scraping yields no results
        urls = google_custom_search(query)
    return urls

def get_text(url):
    headers_list = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:40.0) Gecko/20100101 Firefox/40.0'},
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
    ]
    headers = random.choice(headers_list)  # Randomly choose a user agent

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs if p.get_text().strip())
        return text.strip() if text else None
    except Exception as e:
        print(f"Error fetching text from URL: {e}")
        return None

# Read content from uploaded files
def read_text_file(file):
    return file.getvalue().decode('utf-8', errors='ignore')  # Ignore errors for malformed characters

def read_docx_file(file):
    try:
        return docx2txt.process(file)
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""  # Return empty string if an error occurs

def read_pdf_file(file):
    try:
        text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensure we don't append None if extraction fails
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""  # Return empty string if an error occurs

def get_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return read_text_file(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        return read_pdf_file(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx_file(uploaded_file)
    else:
        print(f"Unsupported file type: {uploaded_file.type}")
        return ""  # Return empty string if unsupported file type

# Compute cosine similarity between two text documents
def get_similarity(text1, text2, cv=None):
    # Preprocess text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Reuse CountVectorizer if passed
    if cv is None:
        cv = CountVectorizer()
        count_matrix = cv.fit_transform([text1, text2])
    else:
        count_matrix = cv.transform([text1, text2])

    # Return cosine similarity
    return cosine_similarity(count_matrix)[0][1]

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list

# Compare text with multiple URLs
def get_similarity_list2(text, url_list):
    similarity_list = []
    for url in url_list:
        if url:
            try:
                text2 = get_text(url)
                if text2:
                    similarity = get_similarity(text, text2)
                    similarity_list.append((url, similarity))
                else:
                    print(f"No text extracted from URL: {url}")
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
        
        # Stop if we've already collected 5 results
        if len(similarity_list) >= 10:
            break
    
    return similarity_list[:5]

def plot_scatter(df):
    # Update the plot to reflect the actual column names in df
    fig = px.scatter(df, x='URL', y='Similarity', title='Similarity Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df):
    fig = px.line(df, x='File 1', y='File 2', color='Similarity', title='Similarity Line Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df):
    fig = px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart')
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app configuration
st.set_page_config(page_title='Plagiarism Detection')
st.title('Plagiarism Detector')

st.write("""
### Enter the text or upload a file to check for plagiarism or find similarities between files
""")
option = st.radio(
    "Select input option:",
    ('Enter text', 'Upload file', 'Find similarities between files')
)

if option == 'Enter text':
    text = st.text_area("Enter text here", height=150)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    if uploaded_file is not None:
        text = get_text_from_file(uploaded_file)
        uploaded_files = [uploaded_file]
    else:
        text = ""
        uploaded_files = []
else:
    uploaded_files = st.file_uploader("Upload multiple files (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts = []
    filenames = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text = get_text_from_file(uploaded_file)
            texts.append(text)
            filenames.append(uploaded_file.name)
    text = " ".join(texts)

# Creating the clickable button with custom style
if st.button('Check for plagiarism or find similarities'):
    st.write("### Checking for plagiarism or finding similarities...")
    if not text:
        st.write("### No text found for plagiarism check or finding similarities.")
        st.stop()

    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity'])
        df = df.sort_values(by=['Similarity'], ascending=False)
        # Display the similarity table
        st.write("### Similarity between files")
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Plotting interactive graphs
        plot_line(df)
        plot_bar(df)
    else:
        sentences = get_sentences(text)
        url_list = []
        for sentence in sentences:
            url_list.extend(get_url(sentence))

        if not url_list:
            st.write("### No plagiarism detected. No URLs found.")
            st.stop()

        similarity_list = get_similarity_list2(text, url_list)

        # Sort similarity list by similarity score
        similarity_list.sort(key=lambda x: x[1], reverse=True)

        # Display the similarity table
        st.write("### Plagiarism Detection Results")

        # Create a DataFrame for display
        df = pd.DataFrame(similarity_list, columns=["URL", "Similarity"])

        # Convert URLs into clickable links
        df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

        # Display the DataFrame as an HTML table with clickable links
        st.write(
            df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        # Check if any similarity value exceeds the threshold
        plagiarism_threshold = 0.5
        is_plagiarized = any(similarity >= plagiarism_threshold for _, similarity in similarity_list)

        # Display plagiarism status
        if is_plagiarized:
            st.write("### 🚨 The text is **Plagiarized**!")
        else:
            st.write("### ✅ No plagiarism detected.")


        # Display Plots
        df = pd.DataFrame(similarity_list, columns=["URL", "Similarity"])
        plot_scatter(df)
