import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

candidate_data = pd.read_csv(r"C:\Users\Braintech\Desktop\industry candidates.csv")
input_data = pd.read_csv(r"C:\Users\Braintech\Desktop\industry requirements.csv")

columns_to_match = ['Industry', 'Activity']

candidate_data[columns_to_match] = candidate_data[columns_to_match].fillna('').astype(str)
input_data[columns_to_match] = input_data[columns_to_match].fillna('').astype(str)

similarity_scores = []
vectorizer = TfidfVectorizer()

for index, row in input_data.iterrows():
    input_vectorized_data = vectorizer.fit_transform(row[columns_to_match])
    candidate_vectorized_data = vectorizer.transform(candidate_data[columns_to_match].apply(lambda x: ' '.join(x), axis=1))
    similarity_matrix = cosine_similarity(input_vectorized_data, candidate_vectorized_data)
    similarity_scores.extend(similarity_matrix.mean(axis=0))  

ranked_candidates = pd.DataFrame({
    'Candidate_ID': candidate_data['Job Id'],
    'Industry': candidate_data['Industry'],
    'Location': candidate_data['Location'],
    'Activity': candidate_data['Activity'],
    'Similarity_Score': similarity_scores  
})

ranked_candidates = ranked_candidates.sort_values(by='Similarity_Score', ascending=False).head(10)

print(ranked_candidates)
