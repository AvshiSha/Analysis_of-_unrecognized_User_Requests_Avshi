# Clustering_Unrecognized_User_Requests

This project groups similar user requests together to quickly reveal common themes. It uses simple natural language processing (NLP) techniques to compare sentences and cluster those with similar meanings, even if you’re not an NLP expert.

## Features
- Reads user requests from a CSV file.
- Converts each request into a numerical format using a pre-trained model.
- Group similar requests based on their meaning.
- Selects a few representative requests from each group.
- Assigns an intuitive name to each cluster based on common phrases.
- Saves the results in a JSON file.
- Optionally compares the results with a reference solution.

## Installation
### Requirements
Make sure you have Python installed. Then, install the necessary packages:
```bash
pip install numpy sentence-transformers scikit-learn nltk
```
Download the NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage
1. Create a configuration file (`config.json`) with the following structure:
   ```json
   {
       "data_file": "requests.csv",
       "output_file": "output.json",
       "num_of_representatives": 3,
       "min_cluster_size": 10,
       "example_solution_file": "example_solution.json"
   }
   ```
2. Run the project:
   ```bash
   python main.py
   ```

## Example
### Input
Imagine you have a CSV file named `requests.csv` with these contents:
```
id, request
1. How do I activate my card?
2. I need help activating my card.
3. How can I change my PIN?
4. What is the process to reset my passcode?
```

### Output
After running the script, you might get an output JSON file (`output.json`) that looks like this:
```json
{
  "cluster_list": [
    {
      "cluster_name": "activate my card",
      "requests": [
        "How do I activate my card?",
        "I need help activating my card."
      ],
      "representatives": [
        "How do I activate my card?"
      ]
    },
    {
      "cluster_name": "change my pin",
      "requests": [
        "How can I change my PIN?"
      ],
      "representatives": [
        "How can I change my PIN?"
      ]
    }
  ],
  "unclustered": [
    "What is the process to reset my passcode?"
  ]
}
```

Now you can run the project on your data and see how it clusters similar requests!

---
