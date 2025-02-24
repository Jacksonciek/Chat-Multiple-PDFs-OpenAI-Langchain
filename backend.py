# import os
# import weaviate
# from dotenv import load_dotenv
# from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Weaviate
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from PyPDF2 import PdfReader

# client = weaviate.Client("http://localhost:8080")

# try :
#     # Check if class already exists before creating
#     if "Document" not in [cls["class"] for cls in client.schema.get()["classes"]]:
#         class_schema = {
#             "class": "Document",
#             "vectorizer": "none",
#             "properties": [
#                 {"name": "text", "dataType": ["string"]}
#             ]
#         }
#         client.schema.create_class(class_schema)

#     def comp_process(apikey, pdfs, question):
#         os.environ["OPENAI_API_KEY"] = apikey
#         llm = OpenAI(temperature=0, openai_api_key=apikey)
        
#         text = ""
        
#         for file in pdfs:
#             pdf_reader = PdfReader(file)
#             for page in pdf_reader.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + "\n"
        
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_text(text=text)

#         # Load OpenAI embeddings
#         embeddings = OpenAIEmbeddings()

#         # Initialize Weaviate as Vector Store
#         vectorstore = Weaviate.from_existing_index(
#             client=client,
#             index_name="Document",
#             embedding=embeddings,
#             text_key="text"
#         )

#         # Add chunks to Weaviate (only if not already indexed)
#         for chunk in chunks:
#             vector = embeddings.embed_query(chunk)
#             client.data_object.create({"text": chunk}, "Document", vector=vector)

#         # Query Weaviate for relevant documents
#         docs = vectorstore.similarity_search(question, k=3)

#         # Run question answering
#         read_chain = load_qa_chain(llm=llm)
#         answer = read_chain.run(input_documents=docs, question=question)

#         return answer
# finally:
#     client.close()
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

def comp_process(pdfs, question):
    load_dotenv()
    llm = OpenAI()
    
    text = ""
    
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)
    
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2",
    #     multi_process=True,
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    docsearch = Chroma.from_texts(chunks, embedding=embeddings).as_retriever()
    
    if question:
        docs = docsearch.get_relevant_documents(question)
        read_chain = load_qa_chain(llm=llm)
        answer = read_chain.run(input_documents=docs, question=question)
    
    return(answer) 