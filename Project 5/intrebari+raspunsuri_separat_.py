import re
import streamlit as st
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

# LLM CONFIGURATION
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

# MEMORY CONFIGURATION
conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# PDF FUNCTION
def process_pdfs(files):
    pdf_texts = []
    for file in files:
        pdf_reader = PdfReader(file)
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                pdf_texts.append({"page": page_number + 1, "text": text})
    return pdf_texts

# RAG FUNCTION
def build_rag_chain(pdf_texts):
    texts_with_pages = []
    for entry in pdf_texts:
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(entry['text'])
        for chunk in chunks:
            texts_with_pages.append({"page": entry['page'], "text": chunk})

    vectorstore = Chroma.from_texts([text['text'] for text in texts_with_pages], HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()

    # PROMPT
    system_prompt = '''
        You are a specialized assistant designed to generate test questions for student exams or tests.
        I want under the question, the answer and the number corespondent to the question (ex: "Answer 1"), and for multiple-choice questions, provide options labeled A, B, C, and D.
        Before each question you must give the number of the questions (ex: "Question 1")
        If it is only one question you also must have the number of the question (ex: "Question 1") 
        When you generate the answer, write down the page number where the answer is in the pdf file.
        \n\n
        {context}
        {chat_history}
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    llmchain = LLMChain(llm=llm, prompt=prompt, memory = conversational_memory)
    rag_chain = create_retrieval_chain(retriever, llmchain)
    return rag_chain, texts_with_pages

# Funcție pentru extragerea întrebărilor, opțiunilor și răspunsurilor corecte
def extract_generated_questions_options():
    questions = ""
    for message in st.session_state.chats[st.session_state.active_chat]:
        if message['role'] == 'assistant':
            # Combine questions and options together
            # Extragerea întrebărilor și opțiunilor de răspuns corelate
            question_blocks = re.findall(r'(Question \d+.*?\n(?:[A-D]\) .*?\n)+)', message['content'], re.DOTALL)

            for block in question_blocks:
                questions += f"{block.strip()}\n\n"  # Adaugă fiecare bloc de întrebare și opțiuni
    return questions

# Function to extract generated answers
def extract_generated_answers():
    answers = ""
    for message in st.session_state.chats[st.session_state.active_chat]:
        if message['role'] == 'assistant':
            # Extract answers from the response using regex
            matches = re.findall(r'Answer \d+.*', message['content'])
            for match in matches:
                answers += f"{match.strip()}\n"

    return answers

# Streamlit CONFIGURATION
st.set_page_config(layout="wide")


# INITIALISING SESSION STATES   
if 'chats' not in st.session_state:
    st.session_state.chats = {'Chat 1': []}
if 'active_chat' not in st.session_state:
    st.session_state.active_chat = 'Chat 1'
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = []

# Sidebar 
with st.sidebar:
    st.title("Test Question Generator")
    with st.expander("Menu"):
        if st.button("New chat"):
            new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.chats[new_chat_name] = []
            st.session_state.active_chat = new_chat_name

        if st.button("Delete chat"):
            if len(st.session_state.chats) > 1:
                del st.session_state.chats[st.session_state.active_chat]
                st.session_state.active_chat = list(st.session_state.chats.keys())[0]
            else:
                st.warning("Cannot delete the last remaining chat.")
                
    with st.expander("Chats"):
        
        for chat_name in st.session_state.chats.keys():
            if st.button(chat_name):
                st.session_state.active_chat = chat_name

    with st.expander("Download options"):
    # Butoane pentru descărcarea întrebărilor și opțiunilor de răspuns
        st.download_button(
            label="Download Generated Questions and Options",
            data=extract_generated_questions_options(),
            file_name=f"{st.session_state.active_chat}_questions_with_options.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Generated Answers",
            data=extract_generated_answers(),
            file_name=f"{st.session_state.active_chat}_answers.txt",
            mime="text/plain"
        )

# MAIN FUNCTION
def main():
    #PDF UPLOADING
    with st.form("test_form"):
            uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            if uploaded_files:
                st.session_state.pdf_texts = process_pdfs(uploaded_files)
                st.success("PDF content loaded successfully!")
            # FORM GENERATION
            with st.expander("Configure your test questions!"):
                    num_questions = st.number_input(
                        "Enter the number of questions:", min_value=1, step=1
                    )
                    difficulty = st.selectbox(
                        "Select the difficulty level:", ["Easy", "Medium", "Hard"]
                    )
                    submitted = st.form_submit_button("Generate Test")

    if submitted:
        if not st.session_state.pdf_texts:
            st.warning("Please upload a PDF file before generating the test.")
        else:
            user_input = f"Generate {num_questions} {difficulty} questions based on the uploaded PDFs."
            st.session_state.chats[st.session_state.active_chat].append({"role": "user", "content": user_input})
            st.session_state.rag_chain, texts_with_pages = build_rag_chain(st.session_state.pdf_texts)
            response = st.session_state.rag_chain.invoke({"input": user_input}) ['answer'] ['text']
            st.session_state.chats[st.session_state.active_chat].append({"role": "assistant", "content": response})

    
    # CHAT HISTORY
    # st.write("Chat History:")
    for elem in st.session_state.chats[st.session_state.active_chat]:
        with st.chat_message(elem['role']):
            st.write(elem['content'])

    # Chat input at the bottom of the screen
    user_input = st.chat_input('Insert text here.', key="main_chat_input")

    if user_input:
        st.session_state.chats[st.session_state.active_chat].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = st.session_state.rag_chain.invoke({"input": user_input}) ['answer'] ['text']
            message_placeholder.markdown(full_response)
        
        st.session_state.chats[st.session_state.active_chat].append({"role": "assistant", "content": full_response})

        # Update conversation memory
        conversational_memory.chat_memory.add_user_message(user_input)
        conversational_memory.chat_memory.add_ai_message(full_response)

    

if __name__ == "__main__":
    main()