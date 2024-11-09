import os
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Process

import scipy
import torch
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer

from scalingfilter.utils import get_files


def worker_process(file_paths, column, model_name, batch_size, gpu_id, output_dir):
    def get_first_available_key(dictionary, keys, default=None):
        for key in keys:
            if key in dictionary:
                return dictionary[key]
        return default

    model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}")
    model.eval()
    model = model.half()

    all_texts = []

    with jsonlines.open(os.path.join(output_dir, f"texts_{gpu_id}.jsonl"), "w") as fo:
        for file_path in file_paths:
            try:
                with jsonlines.open(file_path, "r") as f:
                    for idx, data in enumerate(f):
                        text = get_first_available_key(
                            data, [column, "text", "raw_content"], None
                        )
                        if text is None:
                            continue
                        all_texts.append(text)
                        fo.write({"id": f"{file_path}:{idx}", "text": text})
            except Exception as e:
                print(f"Error processing file: {file_path}, Error: {e}")

        print(f"Processing {len(all_texts)} texts...")
        embeddings = model.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    np.save(
        os.path.join(output_dir, f"embeddings_{gpu_id}.npy"), embeddings
    )  # (num_texts, 768)


# borrowed from https://github.com/vertaix/Vendi-Score
def get_semantic_diversity(K, q=1, p=None, normalize=False):
    def weight_K(K, p=None):
        if p is None:
            return K / K.shape[0]
        else:
            return K * np.outer(np.sqrt(p), np.sqrt(p))

    def normalize_K(K):
        d = np.sqrt(np.diagonal(K))
        return K / np.outer(d, d)

    def entropy_q(p, q=1):
        p_ = p[p > 0]
        if q == 1:
            return -(p_ * np.log(p_)).sum()
        if q == "inf":
            return -np.log(np.max(p))
        return np.log((p_**q).sum()) / (1 - q)

    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    if type(K_) == scipy.sparse.csr.csr_matrix:
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def cache_exists(output_dir):
    return os.path.exists(os.path.join(output_dir, "embeddings_0.npy"))


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a[np.newaxis, :]

    if len(b.shape) == 1:
        b = b[np.newaxis, :]

    a_norm = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    return np.dot(a_norm, b_norm.T)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--column", default="text")
    parser.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--load_from_cache", action="store_true")
    args = parser.parse_args()

    print(f"Input dir: {args.input_dir}")

    mp.set_start_method("spawn")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_from_cache and cache_exists(args.output_dir):
        print("Loading embeddings from cache...")
        all_embeddings = []
        for file in os.listdir(args.output_dir):
            if file.endswith(".npy") and file.startswith("embeddings"):
                all_embeddings.append(np.load(os.path.join(args.output_dir, file)))
    else:
        file_paths = list(get_files(args.input_dir, ext=".jsonl"))
        print(f"Found {len(file_paths)} files")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        chunk_size = len(file_paths) // num_gpus

        processes = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = None if i == num_gpus - 1 else (i + 1) * chunk_size
            p = Process(
                target=worker_process,
                args=(
                    file_paths[start_idx:end_idx],
                    args.column,
                    args.model_name,
                    args.batch_size,
                    i,
                    args.output_dir,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_embeddings = []
        for i in range(num_gpus):
            all_embeddings.append(
                np.load(os.path.join(args.output_dir, f"embeddings_{i}.npy"))
            )

    embeddings = np.vstack(all_embeddings)
    del all_embeddings
    embeddings = embeddings.astype(np.float16)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    diversity_scores = []
    np.random.shuffle(embeddings)
    embeddings = embeddings[:100000]
    embeddings = embeddings.reshape(10, 10000, -1)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(10):
        print(f"Calculating Semantic Diversity {i}...")
        sim_matrix = cos_sim(embeddings[i], embeddings[i])
        sim_matrix = (sim_matrix + 1) / 2
        print(
            f"Sim matrix shape: {sim_matrix.shape}, max: {np.max(sim_matrix)}, min: {np.min(sim_matrix)}"
        )
        np.save(os.path.join(args.output_dir, f"sim_matrix_{i}.npy"), sim_matrix)
        diversity_scores.append(get_semantic_diversity(sim_matrix))

    mean_diversity_score = np.mean(diversity_scores)
    std_diversity_score = np.std(diversity_scores)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for i, score in enumerate(diversity_scores):
            f.write(f"{score}\n")
            print(f"Semantic Diversity {i}: {score}")
        f.write(f"Mean Semantic Diversity: {mean_diversity_score}\n")
        f.write(f"Std Semantic Diversity: {std_diversity_score}\n")
        print(f"Mean Semantic Diversity: {mean_diversity_score}")
        print(f"Std Semantic Diversity: {std_diversity_score}")

if __name__ == "__main__":
    main()
