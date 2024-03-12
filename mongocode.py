import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient


client = MongoClient('localhost', 27017) 
db = client['industry']  
collection_candidates = db['candidates']  
collection_requirements = db['requirements']  

candidate_data = pd.DataFrame(list(collection_candidates.find()))
input_data = pd.DataFrame(list(collection_requirements.find()))

columns_to_match = ['Industry', 'Activity']


candidate_data[columns_to_match] = candidate_data[columns_to_match].fillna('').astype(str)
input_data[columns_to_match] = input_data[columns_to_match].fillna('').astype(str)

similarity_scores = []
vectorizer = TfidfVectorizer()

for index, row in input_data.iterrows():
    input_vectorized_data = vectorizer.fit_transform(row[columns_to_match])
    candidate_vectorized_data = vectorizer.transform(candidate_data[columns_to_match].apply(lambda x: ' '.join(x), axis=1))
    similarity_matrix = cosine_similarity(input_vectorized_data, candidate_vectorized_data)
    similarity_scores.append(similarity_matrix.mean(axis=0))

ranked_candidates = pd.DataFrame({
    'Candidate_ID': candidate_data['Job Id'],
    'Industry': candidate_data['Industry'],
    'Location': candidate_data['Location'],
    'Activity': candidate_data['Activity'],
    'Similarity_Score': max(similarity_scores) 
})

ranked_candidates = ranked_candidates.sort_values(by='Similarity_Score', ascending=False).head()

print(ranked_candidates)
