import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import pypdf
import qdrant_client
from langchain.vectorstores import Qdrant
import os
from dotenv import load_dotenv

partidos = {
    "Partido Popular (PP)": "PP",
    #"Partido Socialista Obrero Español (PSOE)": "PSOE",
    #"VOX (VOX)": "VOX",
    "SUMAR (SUMAR)": "SUMAR",
    "Partido animalista con el medio ambiente (PACMA)":"PACMA"
}

def generar_respuesta(question,partido):

    #load api keys
    load_dotenv()

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    qdrant_url = st.secrets["QDRANT_URL"]

    collection_name = partidos[partido]
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    client = qdrant_client.QdrantClient(qdrant_url,
                                        api_key=qdrant_api_key)
    qdrant=Qdrant(client=client, collection_name=collection_name, embeddings=embeddings_model)

    retriever = qdrant.as_retriever()

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    template = """
    Eres un experto en política española, no tienes opinión, eres totalmente objetivo.
    Responde a la pregunta de forma objetiva basándote en la información del <programa político>. 
    El <programa político> contiene información sobre las medidas que el partido {partido} ha planteado para las próximas elecciones generales.
    Intenta que tu respuesta sea lo más fácil de leer posible, utiliza bullet points si lo ves necesario.

    Pregunta: {question}

    Responde en español.
    =========
    <programa político>:
    {summaries}
    """
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question", "partido"])

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                                     return_source_documents=True, verbose=True,
                                                     chain_type_kwargs=chain_type_kwargs)

    return qa({"question":question, "partido": partido})

#Streamlit
st.set_page_config(page_title="23j")
st.image("https://productomania.io/wp-content/uploads/2023/07/Group-1-2.png", caption=None)
st.header("Los programas electorales al desnudo")
st.caption("Selecciona un partido político y pregunta a la IA qué medidas propone en su programa para la cuestión que más te interese. Te dará una respuesta objetiva basada en el contenido del programa político del partido.")
st.caption("[Desarrollado por [Juan Echeverria](https://www.linkedin.com/in/juan-echeverria-arteaga/)]")
partido = st.selectbox("Elige un partido político", ("Partido Popular (PP)", "SUMAR (SUMAR)", "Partido animalista con el medio ambiente (PACMA)"))#, "VOX (VOX)"))
question = st.text_input("Introduce aquí tu pregunta", placeholder="Ej: ¿Qué medidas proponen para reducir el deficit público?")
if st.button("Preguntar a la IA", use_container_width=True, type="primary"):
    with st.spinner(text="Buscando información..."):
        respuesta = generar_respuesta(question, partido)
        st.success("Aquí tienes la respuesta a tu pregunta")
        st.write(respuesta["answer"])
        pages = []
        for source in respuesta["source_documents"]:
            pages.append(source.metadata["page"] + 1)
        pages = list(set(pages))
        pages_sorted = sorted(pages)
        pages_sorted = [str(page) for page in pages_sorted]
        pages_sorted_str = ", ".join(pages_sorted)
        st.write("Respuesta generada a partir del contenido de las páginas "+pages_sorted_str+" del programa electoral de "+partido+".")
else:
    st.write("")





