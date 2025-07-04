from fastembed import TextEmbedding
import onnxruntime as ort
import numpy as np
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def chunk_texts_with_index(
    data_dir_path, df, chunk_file_name, chunk_size, chunk_overlap, min_text_len
):
    """
    Chunk the abstracts in the dataframe and save the chunks to a pickle file.
    """
    print("CHUNKING START!")
    # Create a text splitter that splits text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    original_indexes = []
    titles = []
    update_dates = []

    # Iterate over the rows in the dataframe and split the text into chunks
    for row in df.itertuples(index=True):
        text = row.abstract
        # Skip empty or very short texts
        if pd.isna(text) or len(str(text)) < min_text_len:
            continue

        # Split the text into chunks
        text_chunks = text_splitter.split_text(str(text))
        chunks.extend(text_chunks)

        # Associate the original index with each chunk
        original_indexes.extend([row.Index] * len(text_chunks))
        titles.extend([row.title] * len(text_chunks))
        update_dates.extend([row.update_date] * len(text_chunks))

    chunks_df = pd.DataFrame(
        {
            "chunk_text": chunks,
            "original_index": original_indexes,
            "chunk_id": range(len(chunks)),
            "title": titles,
            "update_date": update_dates,
        }
    )

    print(f"Created {len(chunks)} chunks from {len(df)} abstracts")

    chunks_df.to_pickle(os.path.join(data_dir_path, chunk_file_name))

    print(f"Saved chunked abstracts to {data_dir_path}")

    return chunks_df


def embed_chunks(
    data_dir_path,
    chunks,
    batch_size,
    save_every,
    save_checkpoints,
    fast_embed_name,
    embeddings_file_name,
):
    """
    Embeds text chunks using a specified text embedding model and saves the embeddings to a file.
    """
    print("EMBEDDING START!")
    # Check for CUDA availability
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    if "CUDAExecutionProvider" in providers:
        print("CUDA is available for text embedding")
    else:
        print(
            "CUDA is not available, please install a GPU-enabled version of onnxruntime-gpu to use CUDA for text embedding. Also ensure that torch is installed with CUDA support."
        )
        return None

    # Create the text embedding model
    embedding_model = TextEmbedding(
        model_name=fast_embed_name,
        batch_size=batch_size,  # This controls how many texts are embedded at once
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Embed the chunks in batches and save checkpoints
    all_embeddings = []
    chunks_list = chunks.tolist()

    for i in tqdm(range(0, len(chunks_list), save_every)):
        checkpoint_chunks = chunks_list[i : i + save_every]
        # The embedding model will internally process these in batches of batch_size
        batch_embeddings = list(embedding_model.embed(checkpoint_chunks))

        # Normalize embeddings here
        batch_embeddings = [vec / np.linalg.norm(vec) for vec in batch_embeddings]

        all_embeddings.extend(batch_embeddings)

        if save_checkpoints:
            # Save checkpoint after each save_every chunks
            if i > 0:
                np.save(
                    os.path.join(data_dir_path, f"embeddings_checkpoint_{i}.npy"),
                    np.array(all_embeddings),
                )

    print("Finished embedding chunks")

    # Save final embeddings
    final_embeddings = np.array(all_embeddings)
    np.save(os.path.join(data_dir_path, embeddings_file_name), final_embeddings)

    print(f"Saved final embeddings to {data_dir_path}")

    return final_embeddings
