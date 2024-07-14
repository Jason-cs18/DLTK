# LLM Cookbook
Large language models (LLMs) have revolutionized the way we interact with AI. Unlike previous NLP models, LLMs are multi-tasking and capable of performing a wide range of tasks, from sentiment analysis to question answering. In future, we may need a single model to handle all daily tasks, such as writing emails, answering questions, and even driving cars.

However, most existing open-source LLMs are built for chatbot instead of domain-specific applications. Thus, we have to retrain them from the scratch for application needs.

In this repository, we provide a set of recipes to help you quickly build your own domain-specfic LLM, ranging from data preparation, pre-train, supervised fine-tuning, serving and LLM agents.

> Why we need to pretrain LLMs instead of fine-tuning?
> Domain knowledge is a basic understanding of text and it is hard to obtain from fine-tuning (e.g., abbreviation). In fine-tuning stage, LLM only learns how to response like human and does not update common knowledge.

## Table of contents
- [LLM Cookbook](#llm-cookbook)
    - [Overview](#overview)
    - [Codebase Selection](#codebase-selection)
    - [Data Preparation](#data-preparation)
    - [Learn domain knowledge via pretrain](#learn-domain-knowledge-via-pretrain)
    - [Align LLMs behaviours with instruction tuning](#align-llms-behaviours-with-instruction-tuning)
    - [Efficiently Serving LLMs](#efficiently-serving-llms)
    - [Make LLMs agent for your applications](#make-llms-agent-for-your-applications)

## Overview
> For Chinese readers, please refer to this [tutorial](https://cloud.tencent.com/developer/article/2315386).

<!-- ![alt text](image.png) -->

![alt text](image-1.png)

## Codebase Selection
Although a lot of LLMs are released, we only focus on open-source codebases which contain pre-train and fine-tuning scripts.
- [litgpt](https://github.com/Lightning-AI/litgpt) ![Github stars](https://img.shields.io/github/stars/Lightning-AI/litgpt.svg) ![Github forks](https://img.shields.io/github/forks/Lightning-AI/litgpt.svg) **highly recommended!**
    - well-structured documentation and support most English LLMs.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) ![Github stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory.svg) ![Github forks](https://img.shields.io/github/forks/hiyouga/LLaMA-Factory.svg) **highly recommended!**
    - well-structured documentation and support most Chinese LLMs/MLLMs. 
- [alignment-handbook](https://github.com/huggingface/alignment-handbook) 
    - well-structured documentation and support Zephyr series.
- [Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese) ![Github stars](https://img.shields.io/github/stars/LlamaFamily/Llama-Chinese.svg) ![Github forks](https://img.shields.io/github/forks/LlamaFamily/Llama-Chinese.svg), [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) ![Github stars](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg) ![Github forks](https://img.shields.io/github/forks/ymcui/Chinese-LLaMA-Alpaca.svg), [Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3) ![Github stars](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca-3.svg) ![Github forks](https://img.shields.io/github/forks/ymcui/Chinese-LLaMA-Alpaca-3.svg)
    - well-structured documentation and support Llama series (chinese).

- [CPM-Bee](https://github.com/OpenBMB/CPM-Bee) ![Github stars](https://img.shields.io/github/stars/OpenBMB/CPM-Bee.svg) ![Github forks](https://img.shields.io/github/forks/OpenBMB/CPM-Bee.svg)
    - well-structured documentation and support CPM-Bee series (chinese).
- [TigerBot](https://github.com/TigerResearch/TigerBot) ![Github stars](https://img.shields.io/github/stars/TigerResearch/TigerBot.svg) ![Github forks](https://img.shields.io/github/forks/TigerResearch/TigerBot.svg)
    - well-structured documentation and support tiggerbot-7b/13b/70b (chinese, from llama series).
- [MedicalGPT](https://github.com/shibing624/MedicalGPT) ![Github stars](https://img.shields.io/github/stars/shibing624/MedicalGPT.svg) ![Github forks](https://img.shields.io/github/forks/shibing624/MedicalGPT.svg) **highly recommended!**
    - well-structured docuementation and support diverse Chinese LLMs 
- [wisdomInterrogatory](https://github.com/zhihaiLLM/wisdomInterrogatory) ![Github stars](https://img.shields.io/github/stars/zhihaiLLM/wisdomInterrogatory.svg) ![Github forks](https://img.shields.io/github/forks/zhihaiLLM/wisdomInterrogatory.svg)
    - detailed instructions on dataset construction.

## Data Preparation

Knowledge based construction refer to [wisdomInterrogatory](https://github.com/zhihaiLLM/wisdomInterrogatory). Diversity is a key for pre-training.

- Pretrain / Continual Pretrain (1M~5M tokens): domain-specific data (30%-40%) + opensource data (60%-70%).
- Supervised Fine-tuning (~50K instruction-question-answer): domain-specific (100%).
- Reinforcement Learning (~50K instruction-question-answer): domain-specific (100%).

> _Note: domain-specific data of the 2nd and 3rd stages are instruction data which consists of task description, question and answer. It is created by human experts or LLM itself. If you want to build instructuon data with GPT4, please refer to [Self-Instruct](https://github.com/yizhongw/self-instruct)._

![](https://github.com/yizhongw/self-instruct/raw/main/docs/pipeline.JPG)
_image source: [Self-Instruct](https://github.com/yizhongw/self-instruct)_

### Knowledge Graph
> Why knowledge graph is important? 
> LLMs are often synthesis factual errors due to hallucination. It is because that LLMs lacks real-world understanding. To insert real-world knowledge into LLMs, knowledge graph is a good choice since it can organizes raw text data (e.g., xxx PDFs) to a easy-to-query and easy-to-understand format.

**An standard pipepline to integrate knowledge graph with LLM**

1. Build a automated knowledge graph from raw text data.
2. Build a simple QA with the created knowledge graph.
3. Integrated knowledge graph with LLM QA pipeline

![](https://github.com/Xu1Aan/KGExplorer/raw/main/asset/%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6.png)
*image source: [KGExplorer (Chinese)](https://github.com/Xu1Aan/KGExplorer)*

Although creating a knowledge graph usually involves specialized and complex tasks, we can leverage LLM to build it automatically.

- [Knowledge Graphs Guides in LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/graphs/)
- [Building a Knowledge Graph with LlamaIndex](https://siwei.io/graph-enabled-llama-index/knowledge_graph_query_engine.html)
- [Knowledge Graph Index](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/)
- [Knowledge Graph Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/#step-1-load-data-from-wikipedia-for-guardians-of-the-galaxy-vol-3)
- [Build a Knowledge Graph with LlamaParse and integrate with RAG pipeline (Chinese)](https://segmentfault.com/a/1190000044890510)

![](https://github.com/siwei-io/talks/assets/1651790/495e035e-7975-4b77-987a-26f8e1d763d2)
_image source: [Knowledge Graph Building with LLM](https://colab.research.google.com/drive/1tLjOg2ZQuIClfuWrAC2LdiZHCov8oUbs)_

### Tools
1. [marker: Convert PDF to markdown quickly with high accuracy](https://github.com/VikParuchuri/marker)
2. [token count: counts the number of tokens in a text string, file, or directory](https://github.com/felvin-search/token-count)
3. [Awesome-LLMs-Datasets](https://github.com/lmmlzn/Awesome-LLMs-Datasets)
4. [LLM Datasets: High-quality datasets, tools, and concepts for LLM fine-tuning](https://github.com/mlabonne/llm-datasets)
5. [NeMo-Curator Public: Scalable toolkit for data curation](https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file) **Highly recommended!**
6. [Data-Juicer: A One-Stop Data Processing System for Large Language Models](https://github.com/modelscope/data-juicer) **Higly recommended!**

## Learn domain knowledge via pretrain

Pretrain aims to learn common knowledge and trains LLM in a next-word prediction task.

## Align LLMs behaviours with instruction tuning

Instruction tuning (also called supervised finetuning) targets to make LLMs output in a expected format. The training data is often like this: 
```markdown
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response: 
```

## Efficiently Serving LLMs 
Serving LLMs efficiently employs a lot of popular techniques, such as quantization, KV cache management, and distributed inference. Fortunately, most open-source serving frameworks have supported them. Usually, we choose [vLLM](https://docs.vllm.ai/en/latest/index.html) for throughput-sensitive scenarios and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for latency-sensitive scenarios.

## Make LLMs agent for your applications
A single LLM only acts as a chatbot and answers user questions. To make LLM userful, we need to teach LLM use tools and external knowledge.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

*image source: [Agents and tools](https://huggingface.co/docs/transformers/agents)*

### Tutorials
1. [Agent and tools](https://huggingface.co/docs/transformers/agents), HuggingFace
2. [Advanced RAG on Hugging Face documentation using LangChain](https://huggingface.co/learn/cookbook/advanced_rag), HuggingFace
3. [Build an agent with tool-calling superpowers 🦸 using Transformers Agents](https://huggingface.co/learn/cookbook/agents), HuggingFace
4. [Agentic RAG: turbocharge your RAG with query reformulation and self-query! 🚀](https://huggingface.co/learn/cookbook/agent_rag), HuggingFace

### Practice
1. [Build an agent with LangChain](https://python.langchain.com/v0.1/docs/modules/agents/), LangChain
2. [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), UC Berkeley
3. [Build Multi-Agent Applications with AutoGen](https://microsoft.github.io/autogen/docs/Getting-Started/), Microsoft