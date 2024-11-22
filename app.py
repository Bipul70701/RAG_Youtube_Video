import os
from dotenv import load_dotenv
from pytubefix import YouTube
import whisper
import tempfile
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser



def main():
    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    st.set_page_config(page_title="Groq LLM SaaS Playground", page_icon="🤖", layout="wide")
    st.title("🤖 Groq LLM SaaS Playground")
    st.sidebar.title("⚙️ Application Settings")
    st.sidebar.markdown("Interact with open-source LLMs using Groq API.")
    

    # Step 1: Prompt for the API Key
    if 'groq_api_key' not in st.session_state:
        st.session_state['groq_api_key'] = ''
        st.session_state['authenticated'] = False

    # Show the API key input only if not authenticated
    if not st.session_state['authenticated']:
        with st.expander("🔑 Enter Your GROQ API Key to Unlock", expanded=True):
            groq_api_key = st.text_input("GROQ API Key", type="password")
            YOUTUBE_VIDEO_URL= st.text_input("Youtube URL")
            if st.button("Unlock Application"):
                if groq_api_key:
                    st.session_state['groq_api_key'] = groq_api_key
                    st.session_state['authenticated'] = True
                    st.session_state['url']=YOUTUBE_VIDEO_URL
                    st.success("API Key authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key.")
        return
    
    YOUTUBE_VIDEO=st.session_state['url']
    # Step 2: Main Chatbot Interface
    st.sidebar.subheader("🤖 Chat Settings")

    # Model selection with more options
    m = st.sidebar.selectbox(
        'Choose a model',
        [ 'llama3-8b-8192','gemma-7b-it','gemma2-9b-it',
        'llama3-groq-70b-8192-tool-use-preview','llama3-groq-8b-8192-tool-use-preview',
        'distil-whisper-large-v3-en','llama-3.1-70b-versatile','llama-3.1-8b-instant','llama-3.2-11b-text-preview',
        'llama-3.2-11b-vision-preview','llama-3.2-1b-preview','llama-3.2-3b-preview','llama-3.2-90b-text-preview',
        'llama-3.2-90b-vision-preview','llama-guard-3-8b','llama3-70b-8192','whisper-large-v3','whisper-large-v3-turbo',
        'llava-v1.5-7b-4096-preview','mixtral-8x7b-32768']

    )
    model=ChatGroq(model_name=m,groq_api_key=st.session_state['groq_api_key'])
    

    # This is the YouTube video we're going to use.
    # YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"

    # YOUTUBE_VIDEO="https://www.youtube.com/watch?v=cdiD-9MMpb0"


    # from langchain_openai.chat_models import ChatOpenAI

    # model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
   

    parser = StrOutputParser()

    

    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    

    # Let's do this only if we haven't created the transcription file yet.
    file_path = os.path.join(os.getcwd(), "transcription.txt")
    if not os.path.exists(file_path):
        youtube = YouTube(YOUTUBE_VIDEO)
        audio = youtube.streams.filter(only_audio=True).first()

        # Let's load the base model. This is not the most accurate
        # model but it's fast.
        whisper_model = whisper.load_model("base")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"Temporary directory created at {tmpdir}")
                file = audio.download(output_path=tmpdir)
                print(f"Downloaded audio to {file}")
                print(f"Transcribing audio from {file}...")
                transcription = whisper_model.transcribe(file,fp16=False)["text"].strip()


                print(transcription)
                if not transcription:
                    print("Error: No transcription was generated.")
                    exit(1)
                
                print(file_path)
                with open(file_path, "w") as file:
                    file.write(transcription)
                print(f"Transcription saved to {file_path}.")
                
        except Exception as e:
            print(f"Error occurred: {e}")

    # with open("transcription.txt") as file:
    #     transcription = file.read()
    loader = TextLoader(file_path)
    text_documents = loader.load()

   


    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)

    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # embeddings = OpenAIEmbeddings()


    

    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)
    chain = (
        {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    
   
   
    st.markdown("## 💬 Chat with the Model")
    user_question = st.text_area("📝 Ask a question:", placeholder="Type your query here...")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    try:
        
        if user_question:
            with st.spinner("Thinking..."):
                response=chain.invoke(user_question)
                message = {"human": user_question, "AI": response}
                st.session_state['chat_history'].append(message)
                st.write("🤖 ChatBot:", response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    st.markdown("### 📜 Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            st.markdown(f"*👤 You*: {message['human']}", unsafe_allow_html=True)
            st.markdown(f"*🤖 AI*: {message['AI']}", unsafe_allow_html=True)

    # Sidebar Buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()

    if st.sidebar.button("Log Out"):
        st.session_state['authenticated'] = False
        st.session_state['groq_api_key'] = ''
        st.session_state['chat_history'] = []
        st.rerun()

    



if __name__ == "__main__":
    main()
