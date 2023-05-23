#https://phoenixnap.com/kb/install-spark-on-ubuntu

import findspark
findspark.init()

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from utils import threshold, sample_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import itertools

'''def cosine_similarity(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))'''


def compute_b_d(matrix, d_star):
    b_d = {}
    for docid, tfidf_row in matrix:
        temp_product_sum = 0
        for pos, tfidf_val in enumerate(tfidf_row):
            temp_product_sum += tfidf_val * d_star[pos]
            if temp_product_sum >= threshold:
                b_d[docid] = pos - 1
                break
    return b_d



def pyspark_APDS(pre_processed_data, workers='*'):


    # Map functuion
    def map_fun(pair):
        docid, tf_idf_list = pair
        res = []
        for id_term in range(0, len(tf_idf_list)):
            if id_term > sc_b_d.value[docid]:
                res.append((id_term, (docid, np.nonzero(tf_idf_list))))
        return res



    # Reduce function
    def reduce_fun(pair):
        term, tf_idf_list = pair
        res = []
        for (id1, d1), (id2, d2) in itertools.combinations(tf_idf_list, 2):
            if term == np.max(np.intersect1d(d1, d2)):#max(list(set(d1) & set(d2))):#np.max(np.intersect1d(d1, d2)):
                sim = cosine_similarity([sc_mat.value[id1]], [sc_mat.value[id2]])[0][0]
                if sim >= sc_treshold.value:
                        res.append((id1, id2, sim))
        return res
    
    '''
    def reduce_fun(pair):
        term, tf_idf_list = pair
        res = []

        for (id1, d1), (id2, d2) in itertools.combinations(tf_idf_list, 2):
            if term in d1.indices and term in d2.indices:
                if np.max(d1.indices[d1.indices == term]) == np.max(d2.indices[d2.indices == term]):
                    sim = cosine_similarity(d1, d2)[0][0]
                    if sim >= sc_treshold.value:
                        res.append((id1, id2, sim))

        return restuple
    '''
    
    
    # Create SparkSession 
    conf = SparkConf().setMaster(f"local[{workers}]") \
        .setAppName("all_pairs_docs_similarity.com") \
        .set("spark.executor.memory", "10g") \
        .set("spark.dgrouppedbyriver.memory", "10g")
    sc = SparkContext(conf=conf)
    
    sc_treshold = sc.broadcast(threshold)

    results = {}
    

    for datasets_name, docs_list in pre_processed_data.items():
        
        print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')
        
        docs_list = sample_dict(docs_list) # Sample a set of documents
        doc_keys = list(docs_list.keys())


        vectorizer = TfidfVectorizer()
        tfidf_features = vectorizer.fit_transform(list(docs_list.values())).toarray() # Get the TF-IDF matrix


        doc_freq = np.sum(tfidf_features > 0, axis=0) # Compute document frequency
        print('doc_freq', doc_freq)

        dec_doc_freq = np.argsort(doc_freq)[::-1] #  Decreasing order of dcoument frequency
        print('dec_doc_freq', dec_doc_freq)

        # Ordered matrix 
        matrix = np.array([row[dec_doc_freq] for row in tfidf_features])
        print('matrix', matrix)

        # Computing the list that will feed into the rdd, list of pairs of (docid, tfidf_list)
        list_pre_rrd = list(zip(range(0, len(tfidf_features)), matrix))
        print('list_pre_rrd', list_pre_rrd)

        d_star = np.max(matrix.T, axis=1) # Computing d*
        print('d_star', d_star)


        sc_mat = sc.broadcast(matrix)


        print('\nComputing b_d')
        sc_b_d = sc.broadcast(compute_b_d(list_pre_rrd, d_star)) # Compute and propagate the b_d
        print(' DONE')



        print('\nRDD creation...')
        rdd = sc.parallelize(list_pre_rrd, numSlices=10*workers) # Creare the RDD
        print(' DONE')



        print('\nAdding flatMap (map_fun) transformation...')
        mapped = rdd.flatMap(map_fun)
        print(' DONE')
        #print('\nDebug Print of the first mapped value')
        #for tuple in mapped.collect(): print(tuple)


        print('\nAdding groupByKey transformation...')
        grouppedby = mapped.groupByKey().mapValues(list)
        print(' DONE')
        #print('\nDebug Print of the first grouppedby value')
        #for tuple in grouppedby.collect(): print(tuple)


        print('\nAdding flatMap (reduce_fun) transformation...')
        reduced = grouppedby.flatMap(reduce_fun).persist()
        print(' DONE')
        #print('\nDebug Print of the first reduced value')
        #print(reduced.first())


        print('\nRunning .collect() action with all transformations')
        start = time.time()
        reduced_results = reduced.collect()
        end = time.time()
        print(' DONE')
        

        print('\nSimilar Documents: ')
        if(len(reduced_results) == 0): print('None')
        for (id1, id2, sim) in reduced_results:
            print(doc_keys[id1], doc_keys[id2], sim)

        results[datasets_name] = {'threshold': threshold, 'uniqie_pairs_sim_docs': len(reduced_results), 'elapsed': end-start}

        
    sc.stop()


    return results
