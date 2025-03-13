# Analysis_of_unrecognized_User_Requests

This project clusters user requests based on semantic similarity using **sentence embeddings** and **cosine similarity metrics**. It further identifies key representatives for each cluster and assigns intuitive names based on common phrases.

## Features
- **Reads requests from a CSV file** and preprocesses them.
- **Generates embeddings** using the `all-MiniLM-L6-v2` SentenceTransformer model.
- **Performs clustering** using cosine similarity with dynamic thresholding.
- **Selects key representatives** within each cluster using a centroid-based greedy approach.
- **Assigns cluster names** based on frequently occurring phrases.
- **Saves the results** in a structured JSON format.
- **Evaluates the clustering results** against a provided reference solution.

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install numpy sentence-transformers scikit-learn nltk
```
Download the NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage
1. Prepare a configuration file (`config.json`) specifying:
   ```json
   {
       "data_file": "requests.csv",
       "output_file": "output.json",
       "num_of_representatives": 3,
       "min_cluster_size": 10,
       "example_solution_file": "example_solution.json"
   }
   ```
2. Run the script:
   ```bash
   python main.py
   ```

## Output
- A **JSON file** containing:
  - Cluster names
  - Clustered requests
  - Representative requests
  - Unclustered requests

