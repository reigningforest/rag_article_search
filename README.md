# rag_article_search


## Data Location
The data is the **[arXiv Dataset in Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)**:

## Prerequisites
### Python download
Ensure you have **[Python](https://www.python.org/downloads/)** installed on your system.

### Package Installation
Install required dependencies by running in terminal:
```
pip install -r requirements.txt
```
NOTE: The `kaggle` and `kagglehub` packages are required for downloading the data.

For optimal GPU acceleration, install the CUDA-enabled version of ONNX Runtime:
```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```
Make sure you install based off of your CUDA version. More info **[here](https://onnxruntime.ai/docs/install/)**.

### Kaggle Account
1. To download the data, you will need to have a Kaggle Account.
2. Obtain the Kaggle API JSON file and paste it into your `.kaggle/` directory
    - Steps are located **[here](https://www.kaggle.com/docs/api#authentication)**.

### API Keys
Please ensure that you have a `.env` file in the project root directory that has:
```
PINECONE_API_KEY=<YOUR API KEY HERE>
OPENAI_API_KEY=<YOUR API KEY HERE>
```

### Directory Setup
All the below code should be run in terminal under the root directory of the project.
```
python example.py
```

Please note that you should only use:
1. A Data folder to store and retrieve data (e.g., `data`)
    - This will be input.
2. A Output folder to store outputs (e.g., `output`)
    - This is automatically generated.

## Usage
### Data Preparation
Note: I have changed the chunking to now include update_date and article title. These were not added to the vector DB as I have only embedded the chunked abstract content. I have also changed the upserting to automatically create a Pinecone index when upserting.

1. Run `download_filter_embed_upsert.py` to download, filter, embed, and upsert data.
    - The following files will be created and stored under the data folder.
        - `arxiv-metadata-oai-snapshot.json`
        - `filtered_abstracts.pkl`
        - `chunked_abstracts.pkl`
        - `embeddings_final.npy`

### Run Automatically
1. Run `batch_processor.py` to read the prompt file and generate outputs.
    - The following outputs will be created and stored under the output folder
        - `output_llm_query.txt`
        - `output_rag_classifier.txt`
        - `output_rag_rewrite.txt`
        - `output_rag_simple.txt`
        - `output_rag_2_advanced.txt`
        - `output_responses.csv`
            - All outputs from the previous txt files, but formatted nicely

### Run Manually
If you do decide to run manually, the outputs will be printed in the terminal. Outputs are not automatically saved.

1. Run `abstract_llm_only.py` to get results from ChatGPT 3.5 Turbo only
    - The question is directly answered by the LLM.
    - You will be prompted to name the data folder

2. Run `abstract_rag_query_simple.py` to get results from ChatGPT 3.5 Turbo using a simple RAG
    - The question is embedded and the top 10 closest embeddings from the VectorDB are used as context for the LLM to generate an answer.
    - You will be prompted to name the data folder

3. Run `abstract_rag_query_classifier.py` to get results from ChatGPT 3.5 Turbo with a classifier to determine whether the RAG is needed or not.
    - The LLM is queried to determine whether the RAG is needed to answer the question.
    - If RAG is needed, the question is embedded and the top 10 closest embeddings from the VectorDB are used as context for the LLM to generate an answer.
    - Otherwise, the question is directly answered by the LLM with no further context added.
    - You will be prompted to name the data folder

4. Run `abstract_rag_rewrite.py` to get results from ChatGPT 3.5 Turbo with a rewrite prior to chunking.
    - The LLM is queried to generate multiple query variations.
    - These query variations then embedded and the top 10 closest embeddings from the VectorDB are used as context for the LLM to generate an answer.
    - You will be prompted to name the data folder

5. Run `abstract_rag_2_advanced.py` to get results from ChatGPT 3.5 Turbo with a classifier to determine whether the RAG is needed or not as well as a rewrite prior to chunking.
    - The LLM is queried to determine whether the RAG is needed to answer the question.
    - If RAG is needed, the LLM is queried to generate multiple query variations, which are embedded, and the top 10 closest embeddings from the VectorDB are used as context for the LLM to generate an answer.
    - Otherwise, the question is directly answered by the LLM with no further context added.
    - You will be prompted to name the data folder

### Optional
1. Run `abstract_db_query.py` to get similar abstracts for a random abstract
    - You will be prompted for a query string.
    - `output_db_query.txt` will be created in the output folder.
