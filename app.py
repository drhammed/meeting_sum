# app.py

from __future__ import print_function
import os
import re
import fitz  # PyMuPDF
import json
import streamlit as st
from docx import Document
import configparser
from GDriveOps.GDhandler import GoogleDriveHandler
import nltk
import string
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
import warnings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import voyageai
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from rouge_score import rouge_scorer
from io import BytesIO
import zipfile  

# Initialize NLTK components
from nltk.stem import WordNetLemmatizer

# Initialize NLTK components
nltk.download('punkt_tab')

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Meeting Summarization",
    layout="wide",
    initial_sidebar_state="expanded",
)

class TranscriptSummarizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        sentences = nltk.sent_tokenize(text)
        punctuation = set(string.punctuation)

        processed_sentences = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            filtered_words = [
                self.lemmatizer.lemmatize(word.lower()) 
                for word in words 
                if word.lower() not in punctuation and word.isalpha()
            ]
            processed_sentences.append(' '.join(filtered_words))

        processed_text = ' '.join(processed_sentences)
        processed_text = re.sub(r'\d+', '', processed_text)

        return processed_text

    def get_model(self, selected_model, GROQ_API_KEY):
        model_mapping = {
            "llama3-8b-8192": "llama3-8b-8192",
            "llama3-70b-8192": "llama3-70b-8192",
            "llama-3.2-1b-preview": "llama-3.2-1b-preview",
            "llama-3.2-3b-preview": "llama-3.2-3b-preview",
            "llama-3.2-11b-text-preview": "llama-3.2-11b-text-preview",
            "llama-3.2-90b-text-preview": "llama-3.2-90b-text-preview"
        }
        if selected_model in model_mapping:
            return ChatGroq(
                groq_api_key=GROQ_API_KEY, 
                model=model_mapping[selected_model], 
                temperature=0.02, 
                max_tokens=500,  
                timeout=60, 
                max_retries=2
            )
        else:
            raise ValueError("Invalid model selected")

    def summarize_text(self, text, selected_model, prompt, GROQ_API_KEY, chunk_size=8000, chunk_overlap=500, similarity_threshold=0.8, num_clusters=10):
        llm_mod = self.get_model(selected_model, GROQ_API_KEY)
        system_prompt = prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        
        conversation = LLMChain(llm=llm_mod, prompt=prompt_template)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        
        # For simplicity, I'll skip embedding and clustering in the prototype
        # Directly send the entire text to the model
        summary = conversation.run({"text": text})
        
        return summary

def main():
    st.title("AI Meeting Summarization with LLaMA 3")
    
    st.markdown("""
    This is a demo on how LLaMA3 can automatically summarize meeting transcripts.
    Upload a `.txt` file containing your meeting transcript, and the AI will generate a concise summary.
    """)
    
    # Sidebar for Configuration
    st.sidebar.header("Configuration")
    
    # API Key Input
    GROQ_API_KEY = st.sidebar.text_input(
        "Groq AI API Key",
        type="password",
        help="Enter your Groq AI API key to access LLaMA 3 model."
    )
    
    # Model Selection
    model_options = [
        "llama3-8b-8192", 
        "llama3-70b-8192", 
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-text-preview",
        "llama-3.2-90b-text-preview"
    ]
    selected_model = st.sidebar.selectbox(
        "Select LLaMA 3 Model",
        options=model_options,
        index=0
    )
    
    # Prompt Template
    default_prompt = "Please provide a concise summary of the following meeting transcript."
    prompt = st.sidebar.text_area(
        "Summarization Prompt",
        value=default_prompt,
        help="Customize the prompt sent to the LLaMA 3 model for summarization."
    )
    
    # Transcript Upload
    st.header("Upload Meeting Transcript")
    uploaded_file = st.file_uploader(
        "Upload a `.txt` file containing the meeting transcript",
        type=["txt"]
    )
    
    # Summarization Button
    if st.button("Generate Summary"):
        if not GROQ_API_KEY:
            st.error("Please enter your Groq AI API key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload a `.txt` transcript file.")
        else:
            # Read the uploaded transcript
            transcript = uploaded_file.read().decode("utf-8")
            
            if not transcript.strip():
                st.error("The uploaded transcript is empty. Please upload a valid `.txt` file.")
            else:
                summarizer = TranscriptSummarizer()
                preprocessed_text = summarizer.preprocess_text(transcript)
                
                if not preprocessed_text.strip():
                    st.error("The preprocessed transcript is empty. Please upload a valid `.txt` file with meaningful content.")
                else:
                    with st.spinner("Generating summary..."):
                        try:
                            summary = summarizer.summarize_text(
                                text=preprocessed_text,
                                selected_model=selected_model,
                                prompt=prompt,
                                GROQ_API_KEY=GROQ_API_KEY,
                                chunk_size=8000,
                                chunk_overlap=500,
                                similarity_threshold=0.8,
                                num_clusters=10
                            )
                            
                            st.subheader("Meeting Summary")
                            st.write(summary)
                            
                            # Optionally, provide a download button for the summary
                            summary_filename = f"Summary-{os.path.splitext(uploaded_file.name)[0]}.txt"
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name=summary_filename,
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"An error occurred during summarization: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed by Hammed Akande. LLM-powered meeting summarization using LLaMA 3 via Groq AI.")

if __name__ == "__main__":
    main()
