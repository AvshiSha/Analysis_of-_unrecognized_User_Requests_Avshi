import csv
import json
import numpy as np
from collections import Counter
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords from NLTK to filter common words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Loads request data from a CSV file
def load_requests(data_file):
    requests = []
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Append each non-empty row
                requests.append(row)

    if not requests:
        print("Error about the data file")  # Log error if file is empty

    return requests


# Generates embeddings for input requests using a pretrained SentenceTransformer model
def generate_embeddings(requests):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(requests, convert_to_numpy=True)


# Clusters requests based on cosine similarity of their embeddings
def perform_clustering(requests, embeddings, min_size, threshold=0.75):
    n = len(embeddings)
    visited = np.zeros(n, dtype=bool)  # Tracks visited points
    clusters_req = {}  # Dictionary to store clustered requests
    clusters_emb = {}  # Dictionary to store clustered embeddings
    unclustered_req = []  # List for requests that do not fit into any cluster
    unclustered_emb = []  # List for their corresponding embeddings
    cluster_id = 0

    # Compute cosine similarity matrix (NxN)
    similarity_matrix = cosine_similarity(embeddings)

    for i in range(n):
        if visited[i]:
            continue  # Skip already visited request

        # Find points similar to the current point based on threshold
        neighbors = np.where(similarity_matrix[i] >= threshold)[0]

        if len(neighbors) < int(min_size):
            continue  # Ignore clusters below min_size threshold
        else:
            j = 0
            for neighbor in neighbors:
                if not visited[neighbor]:
                    j += 1
            if j >= int(
                    min_size):  # If enough neighbors are found, that are not in a different cluster already, create a new cluster
                if cluster_id not in clusters_req:
                    clusters_req[cluster_id] = []
                    clusters_emb[cluster_id] = []
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        clusters_req[cluster_id].append(requests[neighbor])
                        clusters_emb[cluster_id].append(embeddings[neighbor])
                        visited[neighbor] = True
                cluster_id += 1  # Increment cluster ID
            else:
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        unclustered_req.append(requests[neighbor])
                        unclustered_emb.append(embeddings[neighbor])
                        visited[neighbor] = True

    for i in range(n):  # Collect remaining un-clustered requests
        if not visited[i]:
            unclustered_req.append(requests[i])
            unclustered_emb.append(embeddings[i])

    # Compute initial cluster centroids
    cluster_centroids = {cid: np.mean(np.array(embs), axis=0) for cid, embs in clusters_emb.items()}

    max_iterations = 20  # Limit number of re-clustering iterations
    similarity_threshold = 0.712  # similarity threshold for assigning un-clustered points
    for iteration in range(1, max_iterations + 1):
        i = 0
        unclust_to_clust = 0
        still_unclustered_req = []
        still_unclustered_emb = []
        for un_emb in unclustered_emb:  # running on the un-clustered embeddings, try to put them in a new cluster
            sentence = unclustered_req[i]
            best_cluster = None
            best_sim = 0

            # Compare each un-clustered sentence with existing cluster centroids
            for cid, centroid in cluster_centroids.items():
                similarity = cosine_similarity([un_emb], [centroid])[0][0]
                if similarity >= similarity_threshold and similarity > best_sim:
                    best_cluster = cid
                    best_sim = similarity

            if best_cluster is not None:
                clusters_req[best_cluster].append(sentence)
                clusters_emb[best_cluster].append(un_emb)
                unclust_to_clust += 1
            else:
                still_unclustered_req.append(sentence)
                still_unclustered_emb.append(un_emb)

            i += 1

        if unclust_to_clust > 0:
            unclustered_req = still_unclustered_req.copy()
            unclustered_emb = still_unclustered_emb.copy()
        else:
            break  # Stop if no new assignments were made

        # Update cluster centroids
        for cluster_idx in clusters_emb.keys():  # compute the new centroids with the new sentences in each cluster
            cluster_centroids[cluster_idx] = np.mean(np.array(clusters_emb[cluster_idx]), axis=0)

    return clusters_req, unclustered_req, clusters_emb, unclustered_emb


# Selects representative sentences from each cluster based on similarity to centroid
def select_representatives(embeddings, clusters, num_representatives):
    representatives = {}

    for cluster_id, sentences in clusters.items():
        if len(sentences) == 0:
            continue  # Skip empty clusters

        # Convert list of embeddings for this cluster into a NumPy array
        cluster_embs = np.array(embeddings[cluster_id])
        cluster_sentences = sentences

        # Compute the centroid of the cluster
        centroid = np.mean(cluster_embs, axis=0)

        # Compute similarity of each sentence to the centroid
        sims_to_centroid = cosine_similarity(cluster_embs, [centroid]).flatten()

        # Start with the sentence most similar to the centroid
        rep_indices = [int(np.argmax(sims_to_centroid))]

        # Greedily select additional representatives ensuring diversity
        while len(rep_indices) < min(num_representatives, len(cluster_sentences)):
            best_candidate = None
            best_score = -np.inf  # We want to maximize a diversity score

            for i in range(len(cluster_sentences)):
                if i in rep_indices:
                    continue
                candidate_emb = cluster_embs[i]
                # Calculate similarity between candidate and already selected reps
                sims_to_reps = cosine_similarity([candidate_emb], cluster_embs[rep_indices])[0]
                # Lower maximum similarity means the candidate is more diverse.
                # We take negative of the maximum similarity as our diversity score.
                diversity_score = -np.max(sims_to_reps)

                if diversity_score > best_score:
                    best_score = diversity_score
                    best_candidate = i

            if best_candidate is not None:
                rep_indices.append(best_candidate)
            else:
                break  # No candidate found; exit the loop

        # Collect the representative sentences for this cluster
        representatives[cluster_id] = [cluster_sentences[i] for i in rep_indices]

    return representatives


# Assigns names to clusters based on most common phrases in representatives
# This function extracts frequent phrases (trigrams and four-grams) from the representative
# sentences of each cluster and assigns the most common phrase as the cluster name.
def assign_cluster_names(representatives):
    cluster_names = {}
    for cluster_id, reps in representatives.items():
        phrase_counter = Counter()
        for rep in reps:
            tokens = rep.lower().split()  # ensure the tokens are in lower case and clean
            # Extract n-grams from bigrams up to five-grams
            for n in range(2, 6):
                for i in range(len(tokens) - n + 1):
                    candidate = tokens[i:i + n]
                    # Trim stop words from the beginning and end
                    candidate_trimmed = list(candidate)
                    while candidate_trimmed and candidate_trimmed[0] in stop_words:
                        candidate_trimmed = candidate_trimmed[1:]
                    while candidate_trimmed and candidate_trimmed[-1] in stop_words:
                        candidate_trimmed = candidate_trimmed[:-1]
                    # Only consider phrases with at least two words
                    if len(candidate_trimmed) >= 2:
                        phrase = " ".join(candidate_trimmed)
                        phrase_counter[phrase] += 1
        if phrase_counter:
            # Use weighted score: frequency multiplied by the number of words
            best_phrase = max(phrase_counter.items(), key=lambda x: x[1] * len(x[0].split()))
            cluster_names[cluster_id] = best_phrase[0]
        else:
            cluster_names[cluster_id] = reps[0] if reps else "unknown"
    return cluster_names


# Saves clustering results
def save_results(output_file, clusters, unclustered, representatives, cluster_names):
    output_data = {
        "cluster_list": [
            {
                "cluster_name": cluster_names[cluster_id],
                "requests": sentences,
                "representatives": representatives[cluster_id]
            }
            for cluster_id, sentences in clusters.items()
        ],
        "unclustered": unclustered
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    requests = load_requests(data_file)
    requests = requests[1:]
    requests = [req[1].lower() for req in requests]
    requests = [req.lstrip("\n") for req in requests]
    embeddings = generate_embeddings(requests)
    clusters_req, unclustered_req, clusters_emb, unclusters_emb = perform_clustering(requests, embeddings, min_size)
    print("Done with Perform clustering")
    representatives = select_representatives(clusters_emb, clusters_req, int(num_representatives))
    print("Done with select representatives to each cluster")
    cluster_names = assign_cluster_names(representatives)
    save_results(output_file, clusters_req, unclustered_req, representatives, cluster_names)

    pass


if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])
