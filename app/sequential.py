import time
from utils import threshold, sample_dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

#def classic_all_pairs_docs_sim(docs_list: List[str], threshold: float):
def classic_all_pairs_docs_sim(docs_list):
        
    doc_similaritis = []
    vectorizer = TfidfVectorizer()
    
    considered_keys = list(docs_list.keys())
    
    features = vectorizer.fit_transform(list(docs_list.values()))
    
    start = time.time()
    similarities = cosine_similarity(features)
    for doc_1, doc_sims in enumerate(similarities):
        for doc_2, doc_sim in enumerate(doc_sims[(doc_1+1):], start=doc_1+1):
            if doc_sim >= threshold:
                doc_similaritis.append((considered_keys[doc_1], considered_keys[doc_2], doc_sim))
    end = time.time()
    
    
    return doc_similaritis, {'threshold': threshold, 'uniqie_pairs_sim_docs': len(doc_similaritis), 'elapsed': end-start}




def npargwhere_all_pairs_docs_sim(docs_list):
        
    vectorizer = TfidfVectorizer()
    
    considered_keys = list(docs_list.keys())
    
    features = vectorizer.fit_transform(list(docs_list.values()))

    start = time.time()
    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0.0)
    idx_doc_similaritis = np.argwhere(similarities > threshold)
    end = time.time()

    return [
        (considered_keys[similar.tolist()[0]], considered_keys[similar.tolist()[1]], similarities[similar[0], similar[1]])
        for similar in idx_doc_similaritis
    ], {
        'threshold': threshold,
        'uniqie_pairs_sim_docs': len(idx_doc_similaritis) // 2,
        'elapsed': end - start,
    }
    

        
        
    
def squential_APDS(pre_processed_data):
    result_classic = {}
    result_np = {}
    for datasets_name, docs_list in pre_processed_data.items():
        
        #docs_list = sample_dict(docs_list)
        
        print(f'Sequential All Documents Pairs Similarities - {datasets_name} - Classic Version')
        similar_list, stat = classic_all_pairs_docs_sim(docs_list)
        print('Similar documents: ')
        for tuple in similar_list: print(tuple)
        result_classic[datasets_name] = stat
        
        print(f'\nSequential All Documents Pairs Similarities - {datasets_name} - numpy Version')
        similar_list, stat = npargwhere_all_pairs_docs_sim(docs_list)
        print('Similar documents: ')
        for tuple in similar_list: print(tuple)
        result_np[datasets_name] = stat
    return result_classic, result_np