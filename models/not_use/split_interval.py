import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


def split_interval(file_path):

    ## long video load
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # caption extract
    captions = [frames['caption'] for frames in data['frame']]

    # SBERT model load
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # caption embedding 
    caption_embeddings = model.encode(captions)

    # similarity calculate
    similarity_list = []
    for i in range(1, len(captions)):
        similarity = cosine_similarity([caption_embeddings[i-1]], [caption_embeddings[i]])[0][0]
        similarity_list.append(similarity)

    # mean similarity (lag3)
    avg_similarities = []
    window_size = 3

    # Use similarity for previous window_size
    avg_similarities.extend(similarity_list[:window_size])

    # mean similarity calculate
    for i in range(window_size, len(similarity_list)):
        avg_similarity = np.mean(similarity_list[i-window_size:i])
        avg_similarities.append(avg_similarity)

    # diff calculate
    diffs = np.abs(np.diff(avg_similarities))

    # treshold set (Z-Score)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    z_scores = (diffs - mean_diff) / std_diff

    # Z-Score treshold (예: 1.5 이상의 값만 선택)
    # 3
    threshold = 3
    split_indices = np.where(z_scores > threshold)[0]

    # split interval
    interval_segments = []
    start_idx = 0

    for idx in split_indices:
    
        # interval lower bound  
        if idx + 1 - start_idx <= 10:
            continue
        else:
            interval_segments.append((start_idx, idx + 1))  
            start_idx = idx + 1

    if len(avg_similarities) - start_idx <= 10 and interval_segments:
        last_segment_start, _ = interval_segments.pop()  
        interval_segments.append((last_segment_start, len(avg_similarities)))  

    else:
        interval_segments.append((start_idx, len(avg_similarities)))

    for idx, segment in enumerate(interval_segments):
        print(f"Split {idx+1}: {segment[0]} ~ {segment[1]}")

    for idx in split_indices:
        print(f"Diff {idx}:  {similarity_list[idx]}")

    return interval_segments

if __name__ == "__main__":
    # interval_segments = split_interval("/kth/TelVid/data/VIRAT_S_000101.mp4-verbv3.json")
    interval_segments = split_interval("/kth/TelVid/data/VIRAT_S_000004.mp4-verbv3.json")

