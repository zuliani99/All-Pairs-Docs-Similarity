import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def classic_squential_APDS(ds_name, sampled_dict, threshold):
        
    doc_similaritis = []
    vectorizer = TfidfVectorizer()
    
    keys = list(sampled_dict.keys())
    
    features = vectorizer.fit_transform(list(sampled_dict.values()))
    
    start = time.time()
    similarities = cosine_similarity(features)
    for doc_1, doc_sims in enumerate(similarities):
        for doc_2, doc_sim in enumerate(doc_sims[(doc_1+1):], start=doc_1+1):
            if doc_sim >= threshold:
                doc_similaritis.append((keys[doc_1], keys[doc_2], doc_sim))
    end = time.time()
    
    return doc_similaritis, ['classic', ds_name, end-start, threshold, len(doc_similaritis)]




def numpy_squential_APDS(ds_name, sampled_dict, threshold):
        
    vectorizer = TfidfVectorizer()
    
    keys = list(sampled_dict.keys())
    
    features = vectorizer.fit_transform(list(sampled_dict.values()))

    start = time.time()
    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0.0)
    idx_doc_similaritis = np.argwhere(similarities > threshold)
    end = time.time()
    
    return [
        (keys[similar.tolist()[0]], keys[similar.tolist()[1]], similarities[similar[0], similar[1]])
        for similar in idx_doc_similaritis
    ], ['numpy', ds_name, end-start, threshold, len(idx_doc_similaritis) // 2]