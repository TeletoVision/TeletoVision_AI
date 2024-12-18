import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import unicodedata

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from accelerate import Accelerator

from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

CHUNK_SIZE = 256
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "intfloat/e5-large"
K = 3
LLM = "google/gemma-2-9b-it"
QUANTIZATION = "bf16" # "qlora", "bf16", "fp16"
MAX_NEW_TOKENS = 512
PROMPT_TEMPLATE = """
You are an AI visual assistant surveillance operator that can analyze real-time traffic analysis and accident detection.

Respond to user's questions as accurately as possible.
Be careful not to answer with false information.

Using the provided caption information, describe the scene in a detailed manner.
{context}

Question: {question}

Answer:
"""


def process_json(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".frame.[].caption",
        text_content=False,
    )
    docs = loader.load()
    chunks = docs.copy()
    return chunks

def process_total_json(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".frame.[].caption",
        text_content=False,
    )
    docs = loader.load()
    
    video_doc = ""
    for doc in docs:
        video_doc += doc.page_content
        video_doc += '\n'
    
    meta_data = {
        'source' : file_path
    }
    
    chunks = [Document(page_content=video_doc, metadata=meta_data)]
    
    return chunks

def create_vector_db(chunks, model_path=EMBEDDING_MODEL):
    """FAISS DB"""
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db
    
def process_jsons_from_dataframe(full_path, video_name):
    
    json_databases = {}
    
    chunks = process_json(full_path)
    db = create_vector_db(chunks)
    
    # Retriever
    
    retriever_similarity = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': K}
    )

    retriever_mmr = db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': K}
    )

    retriever_bm25 = BM25Retriever.from_documents(chunks)
    
    retriever = EnsembleRetriever(
        retrievers=[retriever_similarity, retriever_mmr, retriever_bm25],
        weights=[0.5, 0.5, 0.5]
    )        
    
    json_databases[video_name] = {
            'db': db,
            'retriever': retriever
    }
    return json_databases

def process_total_json_from_dataframe(unique_paths, base_directory):
    
    total_chunks = []
    
    for path in tqdm(unique_paths, desc="Processing JSONs"):
        
        normalized_path = unicodedata.normalize('NFC', path)
        full_path = os.path.normpath(
            os.path.join(
                base_directory, normalized_path.lstrip('./')
            )
        ) if not os.path.isabs(normalized_path) else normalized_path
        json_title = os.path.splitext(os.path.basename(full_path))[0]
        
        print(f"Processing {json_title}...")
        
        chunks = process_total_json(full_path)

        total_chunks.extend(chunks)
    
    db = create_vector_db(total_chunks)
    # db = load_vector_db(total_chunks)
    
    # Retriever

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': K}
    )
    
    json_databases = {
            'db': db,
            'retriever': retriever
    }
    return json_databases

def setup_llm_pipeline():
    model_id = LLM
    quantization_options = {
        "qlora": {"quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)},
        "bf16": {"torch_dtype": torch.bfloat16},
        "fp16": {"torch_dtype": "float16"}
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        **quantization_options.get(QUANTIZATION, {})
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        # temperature=0.2,
        return_full_text=False,
        max_new_tokens=MAX_NEW_TOKENS,
        # do_sample=True,
    )

    hf = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return hf

def format_docs(docs):
    context = ""
    for doc in docs:
        context += f"frame : {doc.metadata['seq_num']-1}"
        context += '\n'
        context += doc.page_content
        context += '\n'
    return context