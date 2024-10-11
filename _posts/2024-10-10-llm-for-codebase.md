---
layout: post
title:  "Harnessing Local Llama to Process Complete Projects: How I use AI for code suggestions and refactoring my Projects"
author: Gurwinder
categories: [ AI ]
image: assets/images/llama_logo.png
featured: false
hidden: false
---

We’ll walk through a Python script that leverages the LangChain framework to process a codebase, embed the data, and perform queries using LLMs.

Before we jump into the code, let’s first go over the necessary setup to run this example, including installing the required dependencies and configuring Ollama and LangChain.

## **Setting Up Ollama**

Ollama is a powerful platform for working with language models like LLaMA, designed for ease of integration with applications such as LangChain. Here’s how you can set it up:

1. **Install Ollama**  
   First, download and install Ollama based on your operating system. You can find the installation instructions on [Ollama’s official website](https://ollama.com/).

2. **Download and Configure LLaMA Model**  
   After installing Ollama, you can download models like `LLaMA`. Use the following command to get the desired version:

   ```bash
   ollama pull llama3.1
   ```

   This will download the model `llama3.1` to your local machine. You’ll be using this in the code to perform LLM-based queries.

3. **Verify Installation**  
   To ensure Ollama is working properly, you can check the installed models:

   ```bash
   ollama list
   ```

   You should see `llama3.1` or whichever model you downloaded in the list.

## **Setting Up LangChain**

LangChain is the backbone of the codebase processing and querying system. It allows seamless integration between language models, document loaders, and retrievers.

**Install LangChain and Dependencies**

   Install LangChain and its required dependencies via `pip`. You’ll also need `Chroma` for vector storage and `HuggingFace` for embeddings:

   ```bash
   pip install langchain langchain-chroma langchain-huggingface langchain-community
   ```

   Depending on your environment, you may also want to install `TextLoader` and other utility packages:

   ```bash
   pip install langchain-text-splitters
   ```

   This ensures you have all the necessary components for document loading, splitting, and embedding.

## **The Problem**

Imagine working on a large software project with hundreds or thousands of Python files. You're tasked with finding specific information or patterns within this sprawling codebase. Manually searching through files is impractical and time-consuming. Enter LangChain, a framework designed to integrate LLMs with external data sources, making it easier to query and interact with large datasets, such as codebases.

The script provided demonstrates how to:

1. Load and process an entire Python codebase.
2. Split the code into smaller, manageable chunks.
3. Embed the code using an embedding model.
4. Set up a retriever to efficiently search the embedded data.
5. Use an LLM to query the embedded data and answer questions.

Let’s dive into the code and explore how it works.

## **Loading the Codebase**

The first step is to load the codebase into a list of documents. This involves recursively walking through the directories of the repository, filtering out irrelevant files, and loading Python files.

```python
from langchain.document_loaders import TextLoader
import os

root_dir = 'path/to/repo'

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.py') and '/.venv/' not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

print(f'{len(docs)}')
```

In this part, we use Python’s `os.walk` function to traverse the directory tree of the codebase, looking for `.py` files (ignoring virtual environments like `.venv`). We then load the content of these Python files and split them into smaller documents using LangChain’s `TextLoader`. This is essential for embedding the code later on.

## **Splitting the Documents**

Once we have our documents, we need to split them into smaller, manageable chunks. This ensures that when the code is embedded, it is done efficiently without losing context.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=200, chunk_overlap=200
)
texts = python_splitter.split_documents(docs)
len(texts)
```

Here, we use the `RecursiveCharacterTextSplitter` with a chunk size of 200 characters and an overlap of 200 characters to break the code into smaller pieces. The overlap ensures that there’s enough context preserved between chunks to make sense of the code when it's queried.

## **Embedding the Documents**

Embedding is a crucial step where we convert the text into vector representations. These vectors allow us to search for similarities between the query and the embedded documents.

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=2048)
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)
```

In this section, we use the `Chroma` vector store to store the document embeddings. For simplicity, a fake embedding model is used in this example, but you can easily swap in real embeddings such as HuggingFace models. The retriever is then created, enabling us to perform efficient similarity searches with a specific query.

## **Querying the Codebase with LLM**

With the documents embedded, we can now use an LLM to retrieve relevant information based on a query. The LangChain framework supports different types of LLMs and enables history-aware retrieval, which allows for more intelligent searches.

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.ollama import ChatOllama

llm = ChatOllama(model="llama3.1")
```

Here, we initialize the LLM using `ChatOllama`, a hypothetical version of the LLaMA model for this example. You can choose different models depending on your needs.

Next, we define two prompt templates—one for generating a search query and another for answering questions based on retrieved documents:

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation"),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)
```

The first prompt handles generating a search query from the conversation history. The second prompt is responsible for answering questions based on the retrieved documents.

## **Example Query**

Finally, we can query the codebase using a specific question. For example, let’s ask how to write a unit test for this project:

```python
question = "How to write unit test for this project" 
result = qa.invoke({"input": question})
result["answer"] = result["answer"].replace('. ', '.\n')
print(result["answer"])
```

The output will be a detailed answer generated by the LLM, based on the context retrieved from the embedded codebase.

## **Conclusion**

This Python script demonstrates the power of combining LLMs with a framework like LangChain to query large codebases efficiently. With the ability to load, split, embed, and query code, this solution offers developers a robust tool for navigating complex projects, answering questions, and retrieving relevant information from code repositories.

Whether you're working with Python, JavaScript, or any other language, LangChain’s modularity allows you to adapt this approach to your specific needs. The possibilities are vast, from automatically generating documentation to assisting with debugging and code reviews.

---

This updated version includes the setup steps for both Ollama and LangChain, ensuring that readers can run the code without any issues.