# Clustering Unrecognized User Requests

<img align="right" src="https://github.com/user-attachments/assets/d65e9bee-51ff-4ffc-9394-85e9c783f2aa" alt="Logo" width="250" />

This project groups similar user requests together to quickly reveal common themes. It uses simple natural language processing (NLP) techniques to compare sentences and cluster those with similar meanings, even if you’re not an NLP expert.

This project offers a scalable and efficient solution for uncovering the underlying patterns in user requests, providing clear and accessible insights that can drive improvements across customer support, product development, and market research.

## Features

- Reads user requests from a CSV file.
- Converts each request into a numerical format using a pre-trained model.
- Group similar requests based on their meaning.
- Selects a few representative requests from each group.
- Assigns an intuitive name to each cluster based on common phrases.
- Saves the results in a JSON file.

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
1. Create a configuration file (config.json) with the following structure. For example, using the sample input below:
   ```json
   {
       "data_file": "requests.csv",
       "output_file": "output.json",
       "num_of_representatives": 1,
       "min_cluster_size": 1,
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
id,request
1,How do I activate my card?
2,Can I cancel a transfer?
3,How can I change my PIN?
4,What is the process to change my pin number?
5,How can I report a stolen card?
6,I would like to activate my new card.
7,I need help activating my card.

```

### Output
After running the script, you might get an output JSON file (`output.json`) that looks like this:
```json

{
    "cluster_list": [
        {
            "cluster_name": "activate my card?",
            "requests": [
                "how do i activate my card?",
                "i would like to activate my new card.",
                "i need help activating my card."
            ],
            "representatives": [
                "how do i activate my card?"
            ]
        },
        {
            "cluster_name": "cancel a transfer?",
            "requests": [
                "can i cancel a transfer?"
            ],
            "representatives": [
                "can i cancel a transfer?"
            ]
        },
        {
            "cluster_name": "change my pin?",
            "requests": [
                "how can i change my pin?",
                "what is the process to change my pin number?"
            ],
            "representatives": [
                "how can i change my pin?"
            ]
        },
        {
            "cluster_name": "report a stolen",
            "requests": [
                "how can i report a stolen card?"
            ],
            "representatives": [
                "how can i report a stolen card?"
            ]
        }
    ],
    "unclustered": []
}

```

Now you can run the project on your data and see how it clusters similar requests!

---
