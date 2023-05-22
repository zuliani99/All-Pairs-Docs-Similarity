#https://phoenixnap.com/kb/install-spark-on-ubuntu

import findspark
findspark.init()

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from utils import threshold, sample_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import multiprocessing as mp
import concurrent.futures
import time
import itertools
from scipy import sparse


def single_b_d(doc_id, doc_tfidf, d_star, threshold):
    temp_product_sum = 0
    sorted_indices = np.argsort(-1*doc_tfidf)
    for pos, termid in enumerate(sorted_indices):
        temp_product_sum += doc_tfidf[termid] * d_star[termid]
        if temp_product_sum >= threshold:
            return doc_id, sorted_indices[pos - 1]


def parallel_b_d(list_pre_rrd, d_star):
    b_d = {}
    num_processes = mp.cpu_count()  # Get the number of CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
		# Submit the tasks to the executor
        futures = [executor.submit(single_b_d, doc_id, doc_tfidf, d_star, threshold)
				for doc_id, doc_tfidf in list_pre_rrd]

		# Process the results as they complete
        for future in concurrent.futures.as_completed(futures):
            doc_id, termid_minus_1 = future.result()
            if termid_minus_1 is not None:
                b_d[doc_id] = termid_minus_1
    
    return b_d



def pyspark_APDS(pre_processed_data, workers='*'):

    # Map functuion
    def map_fun(pair):
        docid, tf_idf_list = pair
        res = []
        sorted_indices = np.argsort(-1*tf_idf_list.toarray()[0])
        for id_term in sorted_indices:
            if id_term > sc_b_d.value[docid]:
                res.append((id_term, (docid, tf_idf_list)))
        return res

    
    def reduce_fun(pair):
        term, tf_idf_list = pair
        res = []

        for (id1, d1), (id2, d2) in itertools.combinations(tf_idf_list, 2):
            if term in d1.indices and term in d2.indices:
                if np.max(d1.indices[d1.indices == term]) == np.max(d2.indices[d2.indices == term]):
                    sim = cosine_similarity(d1, d2)[0][0]
                    if sim >= sc_treshold.value:
                        res.append((id1, id2, sim))

        return res
    
    '''
    # Reduce function
    def reduce_fun(pair):
        term, tf_idf_list = pair
        res = []
        for id1, d1 in tf_idf_list:
            for id2, d2 in tf_idf_list:
                if term == np.max(np.intersect1d(np.nonzero(d1), np.nonzero(d2))):
                    if cosine_similarity([d1], [d2])[0][0] >= sc_treshold and id1 != id2:
                        res.append((id1, id2, cosine_similarity([d1], [d2])[0][0]))
        return res
    '''
    
    
    # Create SparkSession 
    conf = SparkConf().setMaster(f"local[{workers}]") \
        .setAppName("all_pairs_docs_similarity.com") \
        .set("spark.executor.memory", "10g") \
        .set("spark.driver.memory", "10g")
    sc = SparkContext(conf=conf)
    
    sc_treshold = sc.broadcast(threshold)

    results = {}
    

    for datasets_name, docs_list in pre_processed_data.items():
        
        print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')
        
        docs_list = sample_dict(docs_list)

        # Create the features and columns vectors and list of key value pairs
        vectorizer = TfidfVectorizer()
        tfidf_features = vectorizer.fit_transform(list(docs_list.values()))
        list_pre_rrd = list(zip(list(docs_list.keys()), sparse.csr_matrix(tfidf_features)))
        d_star = np.max(tfidf_features.toarray().T, axis=1)


  
        print('\nComputing b_d')
        sc_b_d = sc.broadcast(parallel_b_d(list(zip(list(docs_list.keys()), tfidf_features.toarray())), d_star))
        print(' DONE')
        
        
        print('\nRDD creation...')
        rdd = sc.parallelize(list_pre_rrd, numSlices=10*workers)
        print(' DONE')

        
        print('\nAdding flatMap (map_fun) transformation...')
        mapped = rdd.flatMap(map_fun)
        print(' DONE')
        #print('\nDebug Print of the first mapped value')
        #print(mapped.first())


        print('\nAdding groupByKey transformation...')
        grouppedby = mapped.groupByKey().mapValues(list)
        print(' DONE')
        #print('\nDebug Print of the first grouppedby value')
        #print(grouppedby.first())
        

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
        for tuple in reduced_results: print(tuple)

        results[datasets_name] = {'threshold': threshold, 'uniqie_pairs_sim_docs': len(reduced_results), 'elapsed': end-start}
    sc.stop()


    return results
