import time
from utils import threshold
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#def classic_all_pairs_docs_sim(docs_list: List[str], threshold: float):
def classic_all_pairs_docs_sim(docs_list):
    count = 0
    doc_similaritis = []
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(docs_list)
    
    
    start = time.time()
    similarities = cosine_similarity(features)
    for doc_1, doc_sims in enumerate(similarities):
        for doc_2, doc_sim in enumerate(doc_sims[(doc_1+1):], start=doc_1+1):
            if doc_sim >= threshold:
                count += 1
                doc_similaritis.append((doc_1, doc_2, doc_sim))
    end = time.time()
    
    
    return doc_similaritis, {'threshold': threshold, 'similar_doc': count, 'elapsed': end-start}


#def npargwhere_all_pairs_docs_sim(docs_list: List[str]):
def npargwhere_all_pairs_docs_sim(docs_list):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(docs_list)


    start = time.time()
    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0.0)
    idx_doc_similaritis = np.argwhere(similarities > threshold)
    end = time.time()

    return [
        (similar.tolist(), similarities[similar[0], similar[1]])
        for similar in idx_doc_similaritis
    ], {
        'threshold': threshold,
        'similar_doc': len(idx_doc_similaritis) // 2,
        'elapsed': end - start,
    }
    
'''return [(similar.tolist(), similarities[similar[0], similar[1]]) for similar in idx_doc_similaritis], \
        {'threshold': threshold, 'similar_doc': int(len(idx_doc_similaritis)/2), 'elapsed': end-start}'''
        
        
    
def squential_APDS(pre_processed_data):
    result_classic = {}
    result_np = {}
    for datasets_name, docs_list in pre_processed_data.items():
        print(f'Sequential All Documents Pairs Similarities - {datasets_name} - Classic Version')
        similar_list, stat = classic_all_pairs_docs_sim(list(docs_list.values()))
        print('Similar documents: ')
        for tuple in similar_list: print(tuple)
        result_classic[datasets_name] = stat
        
        print(f'\nSequential All Documents Pairs Similarities - {datasets_name} - numpy Version')
        similar_list, stat = npargwhere_all_pairs_docs_sim(list(docs_list.values()))
        print('Similar documents: ')
        for tuple in similar_list: print(tuple)
        result_np[datasets_name] = stat
    return result_classic, result_np