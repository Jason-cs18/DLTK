# LlamaIndex

LlamaIndex is a data framework for LLM-based applications which benefit from context augmentation. Such LLM systems have been termed as RAG systems, standing for “Retrieval-Augmented Generation”. LlamaIndex provides the essential abstractions to more easily ingest, structure, and access private or domain-specific data in order to inject these safely and reliably into LLMs for more accurate text generation..

**Official Doc:** https://docs.llamaindex.ai/en/stable/

## RAG Background

[What Is Retrieval Augmented Generation, or RAG? | Databricks](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)

![alt text](https://docs.llamaindex.ai/en/stable/_images/basic_rag.png)

Why we need RAG? (from the application perspective):
    - Maintain up-to-date information.
    - Access domain-specific knowledge.

What we need to do in RAG? (from the implementation perspective):
    - Index relevant data. # vector database
    - Retrieve relevant data. # rerank and query
    - Construct a question/query and send it to the LLM. # prompt engineering

## Installation
`pip install llama-index`

## Pipeline
![alt text](https://docs.llamaindex.ai/en/stable/_images/stages.png)

## Use Cases

## Examples