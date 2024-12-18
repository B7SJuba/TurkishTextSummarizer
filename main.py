#Importing libraries
import numpy as np
#import pandas as pd
#import os
#import tensorflow as tf
#import torch
#import spacy
#import nltk
import mpld3
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.lang.tr.stop_words import STOP_WORDS #The stop words for turkish language
from nltk.tokenize import sent_tokenize  # We need nltk sentence tokenizer to tokenize sentences
#from gensim.models import Word2Vec
#from sklearn.model_selection import train_test_split
#from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
#Some libraries are not used because I tried using another approach but deducted that code to a text file, kept the libraries just in case.



test_text = """İlk olarak Çin’in Wuhan bölgesinde, 2019 yılı Aralık ayının başında görülüp, bu bölgedeki yetkililer tarafından tanımlandığı için gayri resmi Wuhan koronavirüsü adıyla da bilinen yeni koronavirüs solunum yolu enfeksiyonuna neden olan ve insandan insana geçebilen bulaşıcı bir virüstür.

Dünya Sağlık Örgütü (WHO) tarafından virüsün resmi adı SARS-CoV-2 (Şiddetli Akut Solunum Sendromu-Koronavirus-2) olarak belirlenmiştir. Dünya Sağlık Örgütü virüsün neden olduğu hastalığı tanımlamak için COVID-19 terimini kullanmaktadır.

30 Ocak 2020'de CoViD-19, Dünya Sağlık Örgütü tarafından küresel bir sağlık acil durumu ilan edilmiştir. 11 Mart 2020 tarihinde ise virüs pandemi, yani küresel salgın hastalık olarak ilan edilmiştir."""

def textRank (text_to_summarize, N):
    # This function uses the textrank algorithm to summarize text.

    # Inputting an empty array caused a lot of issues, this is a simple workaround it.
    if len(text_to_summarize) == 0:
        text_to_summarize = test_text

    #Importing the pretrained model which will change our sentences into vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Tokenizing the sentences: replacing the data with values that points to it. done using nltk
    sentences = sent_tokenize(text_to_summarize)

    # Preprocessing the text to make the text noise-free as much as possible
    # re.sub() to remove all non-alphanumeric characters (excluding whitespace) from the input string.
    sentences_clean = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]

    print("Sentence clean with stopwords:")
    print(sentences_clean)
    print("--------------------------------------------")

    stopwords = list(STOP_WORDS)  # Taking stop words from spaCy
    # created a function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stopwords])
        return sen_new

    sentences_clean = [remove_stopwords(r.split()) for r in sentences_clean]
    print("Sentence clean without stopwords:")
    print(sentences_clean)
    print("--------------------------------------------")

    # Now we provide the vector representation of the sentences
    # which are encoded by calling model.encode()
    # The model will embedd each sentence with its vector representation using sentenceTransformers pretrained model
    sentence_vectors = model.encode(sentences_clean)


    # Printing the sentences with their embedded vector.
    # "Embedding" generally refers to the process of representing objects, data, or information in a lower-dimensional space, typically with a fixed-size vector
    # Each element of the vector is a floating-point number.
    # Word embeddings are representations of words in a continuous vector space. Words with similar meanings are often close to each other in this space.
    print("Sentence vectors:")
    for sentence, embedding in zip(sentences, sentence_vectors):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print(" ")
    print("--------------------------------------------")

    # Initializing a matrix with zeros before populating it with actual values
    similarity_matrix = np.zeros([len(sentences), len(sentences)])

    # Reshaping all arrays in the list cosine similarity expects a 2D array
    # Reshape function will change the dimension of the array into 2D array, (1, -1) will give us 1 row and as many columns needed.
    reshaped_vectors = [vector.reshape(1, -1) for vector in sentence_vectors]


    # Calculating cosine similarity and populating the similarity matrix
    # the angle between the vectors is an important factor in determining their cosine similarity. The smaller the angle, the larger the cosine similarity.
    # If the vectors are pointing in the same direction, the cosine similarity will be 1.
    # If they are orthogonal (more like perpandicular 90), the cosine similarity will be 0. If they point in exactly opposite directions, the cosine similarity will be -1
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            # To ensure that a sentence is not compared to itself
            if i != j:
                # Calculating the cosine similarity
                # [0, 0] indexing is used to get the cosine similarity value (a single scalar) from that matrix.
                # cause cosine similarity matrix sometimes returns matrix even when dealing with individual pairs.
                similarity_matrix[i, j] = cosine_similarity(reshaped_vectors[i], reshaped_vectors[j])[0, 0]

    print("Similarity matrix:")
    print(similarity_matrix)
    print("Similarity matrix shape:")
    print(similarity_matrix.shape)
    print("--------------------------------------------")

    # Creating a heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(6, 4))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt=".2f", xticklabels=False, yticklabels=False,
                ax=ax_heatmap, annot_kws={"size": 6})
    ax_heatmap.set_title('Similarity Matrix', color='white')
    html_heatmap = mpld3.fig_to_html(fig_heatmap)  # Converting plot to html

    # Converting the similarity matrix into a graph
    nx_graph = nx.from_numpy_array(similarity_matrix)
    print("Graph: ")
    print(nx_graph)
    print("--------------------------------------------")

    # Filtering out weak connections to make sure that only high similarity connections are displayed
    # That fixed the issue I had where every node was connected to the other.
    threshold = 0.7
    filtered_edges = [(i, j) for i, j, w in nx_graph.edges(data='weight') if w > threshold]
    nx_graph_filtered = nx.Graph(filtered_edges)
    print("Filtered Graph: ")
    print(nx_graph_filtered)
    print("--------------------------------------------")

    # Visualizing the graph using a spring layout
    fig_nx, ax_nx = plt.subplots(figsize=(6, 4))
    pos = nx.kamada_kawai_layout(nx_graph_filtered)
    nx.draw(nx_graph_filtered, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=200, ax=ax_nx)
    ax_nx.set_title('Graph Representation from Similarity Matrix', color='white')
    html_nx = mpld3.fig_to_html(fig_nx)

    # Printing nodes
    print("Nodes:", list(nx_graph_filtered.nodes()))

    # Printing edges
    print("Edges:")
    for edge in nx_graph_filtered.edges():
        print(f"- {edge[0]} is connected to {', '.join(map(str, nx_graph_filtered.neighbors(edge[0])))}")
    print("--------------------------------------------")
    # Applying the page rank algorithm
    scores = nx.pagerank(nx_graph)
    print("PageRank Scores: ")
    print(scores)
    print("--------------------------------------------")

    # Creating a bar chart
    #I had to make sure the node_indices are correctly represented on the x-axis and node_probs are used as the heights of the bars.
    # that's why it didn't represent the values correctly.
    fig_scores, ax_scores = plt.subplots(figsize=(6, 4))
    node_indices = list(scores.keys())
    node_probs = list(scores.values())
    ax_scores.bar(node_indices, node_probs, align='center', color='skyblue')
    ax_scores.set_xlabel('Node Index', color='white')
    ax_scores.set_ylabel('PageRank Score', color='white')
    ax_scores.set_title('PageRank Scores of Nodes', color='white')
    html_scores = mpld3.fig_to_html(fig_scores)

    # now we extract the sentences with the highest rank using enumerate(), which allows you to iterate through a sequence and keep track of the index of each element
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = []
    print("Scores Ranked descending: ")
    print(ranked_sentences)
    print("--------------------------------------------")

    # Extracting top N sentences
    # To avoid index getting out of range
    if N > len(ranked_sentences):
        N = len(ranked_sentences)

    for i in range(N):
        summary.append(ranked_sentences[i][1])

    # now we need to combine the sentences together
    final_sentence = [w for w in summary]
    final_summary = ' '.join(final_sentence)

    print("----------------------------------------------------------------")
    print("The summarized text with PageRank: ")
    print("Original:" + text_to_summarize)
    print(" ")
    print("Summary: " + final_summary)
    print("--------------------------------------------")
    print("Original text:", len(text_to_summarize), "Summarized text:", len(final_summary))
    print("----------------------------------------------------------------")

    inputLength = len(text_to_summarize)
    summaryLength = len(final_summary)

    #Returning the values to be shown on the website.
    return final_summary, inputLength, summaryLength, html_heatmap, html_nx, html_scores


#textRank(test_text, 1)




