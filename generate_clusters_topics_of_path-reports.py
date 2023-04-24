import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
from sklearn.cluster import KMeans
from bertopic._ctfidf import ClassTFIDF
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

import os
import glob

#1. read in all the path report txt files, as a list of strings
path_files_texts = []
path_to_dir_of_path_reports = "/path/to/dir/containing/path/reports"

# Loop through all the text files in the directory
for file in glob.glob(os.path.join(path_to_dir_of_path_reports, "*.txt")):
    # Open each file and read its contents into a string
    with open(file, "r") as f:
        text_contents_of_file = f.read()
        # Append the string to the list
        path_files_texts.append(text_contents_of_file)

#2. generate embeddings using cohere's embed endpoint
import cohere

apiKey = 'acYUhCGgSu54zqgZDFrrJtgLsNFYMF59L0992b6I'
co = cohere.Client(apiKey)

embeds = co.embed(texts=path_files_texts,                  				
  					model="small",
  					truncate="START").embeddings


'''
# alternate way to call the embed api using requests
import requests

# Set up the API endpoint and parameters
endpoint = "https://api.cohere.ai/embed/v1"
params = {
    "model": "baseline",
    "input": "Hello, world!"
}

# Set up the authentication header with your API key
headers = {
    "Authorization": "Bearer YOUR_API_KEY_HERE"
}

# Send a POST request to the API endpoint with the parameters and headers
response = requests.post(endpoint, params=params, headers=headers)

# Extract the text embedding from the response
embedding = response.json()["embeddings"][0]

# Print the embedding vector
print(embedding)
'''

#This gives us a matrix embeds(embeddings)
# where each path_report text has a 1024 dimensional vector numerically containing its meaning(as per the 
# cohere "small" model).

#3.
'''
Next, we’ll reduce the embeddings down to two dimensions so we can plot them and 
explore which reports are similar to each other. We use UMAP for this dimensionality reduction.

import umap

reducer = umap.UMAP(n_neighbors=100)
umap_embeds = reducer.fit_transform(embeds)
'''

import umap

reducer = umap.UMAP(n_neighbors=100)
umap_embeds = reducer.fit_transform(embeds)

#import kmeans
from sklearn.cluster import KMeans

# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
#todo
#visualizer.fit(cluster_df)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure



'''
from sklearn.cluster import KMeans

# Pick the number of clusters
n_clusters = 8

# Cluster the embeddings
kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
classes = kmeans_model.fit_predict(embeds)
'''

#todo : how to plot the 8 clusters, (8 : ask hn, 15: all hn posts)
#with a different color for each cluster?

#todo : how to pick the # of clsuters

'''
via twitter/whatsapp conversation with nirantk :

bertopic docs suggests that it uses hdbscan, for the clustering, by default.
https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html

from bertopic import BERTopic
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
topic_model = BERTopic(hdbscan_model=hdbscan_model)


hyperopt
http://hyperopt.github.io/hyperopt/

elbow method
https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

# Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(cluster_df)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure
'''

# 5. Generating *names* of clusters
'''
5.
To extract the main keywords for each cluster, we can use the cTFIDF algorithm from the awesome BERTopic package. That results in these being the main keywords for each cluster:

Each cluster now gets a list of representative words(tokens ?)

https://maartengr.github.io/BERTopic/api/ctfidf.html?ref=txt.cohere.ai#bertopic.vectorizers._ctfidf.ClassTfidfTransformer

'''
from topically import Topically
from bertopic import BERTopic

# Load and initialize BERTopic to use KMeans clustering with 8 clusters only.
cluster_model = KMeans(n_clusters=8)
topic_model = BERTopic(hdbscan_model=cluster_model)

# df is a dataframe. df['title'] is the column of text we're modeling
df['topic'], probabilities = topic_model.fit_transform(df['title'], embeds)

# Naming topics with Topically

# Pass in Cohere API key, or it will ask for it
app = Topically(apiKey)

# Name each cluster. This will make one request to GENERATE for each cluster.
# Since we have 8 topics, this will call Cohere Generate 8 times.
df['topic_name'], topic_names = app.name_topics((df['title'], df['topic']))

# We can see the suggested names of these topics
topic_names