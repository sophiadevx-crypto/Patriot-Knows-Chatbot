import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback

# -------------------------------------------------

import os
import re
import fitz  # pip install pymupdf
import nltk  # pip install nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import inflect # pip install inflect

# Uncomment line below to download the stopwords dataset
#nltk.download('stopwords')
#nltk.download('punkt')

openai_api_key = st.secrets["OPENAI_API_KEY"]

def remove_symbols(text):
    # Remove non-alphanumeric characters (keep only letters and numbers)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def remove_stopwords(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

def remove_specific_words(text, words_to_remove):
    # Remove specific words from the text
    for word in words_to_remove:
        text = re.sub(r'\b' + re.escape(word.lower()) + r'\b', '', text)
    return text

def convert_digits_to_words(text):
    # Convert single-digit numbers to words
    p = inflect.engine()
    words = re.findall(r'\b\d\b', text)
    for digit in words:
        text = text.replace(digit, p.number_to_words(digit))
    return text

def process_user_input(user_input):
    
    # Array of specific words to remove
    specific_words_to_remove = ['dlsud', 'school', 'university', 'code', 'allowed', 'want', 'grade']

    # Remove symbols, remove specific words, remove stopwords, and convert one-digit numbers from user input
    cleaned_input = convert_digits_to_words(remove_stopwords(remove_specific_words(remove_symbols(user_input), specific_words_to_remove)))

    # Tokenize the cleaned input
    words = word_tokenize(cleaned_input)

    # Use inflect to handle both pluralization and singularization
    p = inflect.engine()
    processed_keywords = [p.plural(word) for word in words] + [p.singular_noun(word) or word for word in words]

    # Create phrases with two consecutive words
    phrases = [' '.join(pair) for pair in zip(words, words[1:])]

    # Combine phrases and individual words
    processed_keywords += phrases + words

    # Remove duplicates by converting the list to a set and then back to a list
    processed_keywords = list(set(processed_keywords))

    return processed_keywords

def merge_sort(arr, key_function):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half, key_function)
        merge_sort(right_half, key_function)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if key_function(left_half[i]) > key_function(right_half[j]):
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def binary_search(arr, target, key_function):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if key_function(arr[mid]) == target:
            return mid
        elif key_function(arr[mid]) < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def search_in_pdf(pdf_path, keywords):
    pdf_document = fitz.open(pdf_path)
    mentions_count = defaultdict(int)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text("text").lower()

        # Check if any keyword or phrase is present in the text
        if any(keyword in text for keyword in keywords):
            for keyword in keywords:
                # Use regular expression to match whole words only
                pattern = re.compile(r'\b' + re.escape(keyword.lower()))
                mentions_count[keyword] += len(re.findall(pattern, text))

                # If you want to print occurrences, uncomment the following lines
                matches = re.finditer(pattern, text)
                for match in matches:
                    print(f'Keyword "{keyword}" found in {os.path.basename(pdf_path)} on page {page_num + 1} at position {match.start()}')

    pdf_document.close()
    return mentions_count

def search_in_pdfs(pdf_folder, keywords, top_n=2):
    selected_pdfs_with_phrases = []
    selected_pdfs_without_phrases = []

    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                mentions_count = search_in_pdf(pdf_path, keywords)

                # Count mentions of phrases
                phrase_mentions = sum(mentions_count[phrase] for phrase in keywords if ' ' in phrase)

                total_mentions = sum(mentions_count.values())

                if total_mentions > 0:
                    if phrase_mentions > 0:
                        selected_pdfs_with_phrases.append((pdf_path, mentions_count, phrase_mentions))
                    else:
                        selected_pdfs_without_phrases.append((pdf_path, mentions_count, total_mentions))
                else:
                    pdf_path = './dlsudStudentHandbookAndCBL/Vision and Mission.pdf'
                    selected_pdfs_without_phrases.append((pdf_path, mentions_count, phrase_mentions))

    # if not selected_pdfs_with_phrases and not selected_pdfs_without_phrases:
        # print('No PDFs found matching the criteria.')
        # st.success("Please Enter a Valid Input")
        # return None, None

    # Sort selected PDFs with phrases using merge sort
    merge_sort(selected_pdfs_with_phrases, key_function=lambda x: x[2])

    # Sort selected PDFs without phrases using merge sort
    merge_sort(selected_pdfs_without_phrases, key_function=lambda x: x[2])

    # If there are PDFs with phrases, consider only those in the top_n using binary search
    if selected_pdfs_with_phrases:
        target = selected_pdfs_with_phrases[0][2] if top_n > 0 else float('-inf')
        index = binary_search(selected_pdfs_with_phrases, target, key_function=lambda x: x[2])

        if index != -1:
            top_n_pdfs = selected_pdfs_with_phrases[:min(index + top_n, 2)]
        else:
            top_n_pdfs = selected_pdfs_with_phrases[:2]
    else:
        # If no PDFs with phrases, consider top_n PDFs with individual words
        target = selected_pdfs_without_phrases[0][2] if top_n > 0 else float('-inf')
        index = binary_search(selected_pdfs_without_phrases, target, key_function=lambda x: x[2])

        if index != -1:
            top_n_pdfs = selected_pdfs_without_phrases[:min(index + top_n, 2)]
        else:
            top_n_pdfs = selected_pdfs_without_phrases[:2]

    if not top_n_pdfs:
        pdf_file_path_1 = pdf_file_path_2 = None
    else:
        print(f'The selected PDFs:')
        for i, (pdf_path, mentions, _) in enumerate(top_n_pdfs, 1):
            print(f'Top {i}: {os.path.basename(pdf_path)}')
            # uncomment line below to show what pdfs will process
            # st.write(f"Process: {os.path.basename(pdf_path)}")
            # Set pdf_file_path for the top 1 and top 2 PDFs
            if i == 1:
                pdf_file_path_1 = pdf_path
            elif i == 2:
                pdf_file_path_2 = pdf_path
        # If there is no top 2 but there is a top 1, set pdf_file_path_2 to the same value as pdf_file_path_1
        if len(top_n_pdfs) == 1:
            pdf_file_path_2 = pdf_file_path_1

    return pdf_file_path_1, pdf_file_path_2

# use file path of the folder of pdfs for the pdf_folder variable
pdf_folder = './dlsudStudentHandbookAndCBL/'

# -------------------------------------------------

# Use the st.cache decorator to cache the result of the function
@st.cache_data
def get_pdf_text(pdf_path):
    # If not in cache, extract text from the PDF file
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()

    pdf_reader.stream.close()  # Close the stream explicitly

    # Streamlit cache mechanism will handle the storing of the result in the cache
    print(f"Caching text for the PDF file '{pdf_path}'.")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    load_dotenv()
    # Create llm
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        with get_openai_callback() as cb:
            response = st.session_state.conversation({'question': user_question})
            new_messages = response['chat_history']

            # Initialize chat_history as an empty list if it doesn't exist
            st.session_state.chat_history = st.session_state.chat_history or []

            st.session_state.chat_history.extend(new_messages)
            print(cb)

            # Display chat history
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + '_user')
                else:
                    message(msg.content, is_user=False, key=str(i) + '_ai')

            # st.success("Processing Complete.")
    else:
        # Handle the case when st.session_state.conversation is None
        st.warning("Please click Process first before asking questions.")

def main():
    load_dotenv()
    st.set_page_config(page_title="PATRIOT KNOWS💬")
    st.write(css, unsafe_allow_html=True)
    st.sidebar.image("dlsud_logo.PNG", use_column_width=True)

    st.sidebar.markdown("""
    <div style="text-align: center;">
        <h2 style="margin-bottom: 0; font-size: 24px;">PATRIOT KNOWS</h2>
        <p style="margin: 0; font-size: 16px;">A DLSU-D CBL and Student</p>
        <p style="margin: 0; font-size: 16px;">Handbook Chatbot</p><br><br><br>
    </div>
    """, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.markdown(
        '<div style="font-size: 42px; font-weight: 700; margin-bottom: 20px;">Chat with Patriot Knows💬</div>',
        unsafe_allow_html=True
    )
    st.caption("Ask anything about the Conduct of Blended Learning and the Sections of the Student Handbook :books:")

    user_question = st.chat_input("Your message", key="user_input")

    if user_question:
        keywords = process_user_input(user_question)
        print(f'Processed input keywords: {keywords}')

        top_1_pdf, top_2_pdf = search_in_pdfs(pdf_folder, keywords, top_n=2)
        print(f'{top_1_pdf}')
        print(f'{top_2_pdf}')

        with st.spinner("Processing"):
            raw_text_1 = get_pdf_text(top_1_pdf)

            if top_2_pdf != top_1_pdf:
                raw_text_2 = get_pdf_text(top_2_pdf)
                raw_text = raw_text_1 + raw_text_2
            else:
                raw_text = raw_text_1

            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
            handle_userinput(user_question)

    def reset_conversation():
        st.session_state.conversation = None
        st.session_state.chat_history = None

    st.sidebar.button('Reset Chat', on_click=reset_conversation)


if __name__ == '__main__':
    main()
