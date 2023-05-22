#https://phoenixnap.com/kb/install-spark-on-ubuntu

import findspark
findspark.init()

import pyspark

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from utils import threshold, considered_docs, sample_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import multiprocessing as mp
import concurrent.futures
import time


def single_b_d(doc_id, doc_tfidf, d_star, threshold):
    temp_product_sum = 0
    sorted_indices = np.argsort(-1*doc_tfidf)
    for pos, termid in enumerate(sorted_indices):
        temp_product_sum += doc_tfidf[termid] * d_star[termid]
        if temp_product_sum >= threshold:
            return doc_id, sorted_indices[pos - 1]
    return doc_id, None


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
        sorted_indices = np.argsort(-1*tf_idf_list)
        for id_term in sorted_indices:
            if id_term > b_d[docid]:
                res.append((id_term, (docid, tf_idf_list)))
        return res
    

    
    # Reduce function
    def reduce_fun(pair):
        term, tf_idf_list = pair
        res = []
        for id1, d1 in tf_idf_list:
            for id2, d2 in tf_idf_list:
                if len(np.intersect1d(np.nonzero(d1), np.nonzero(d2))) > 0 and term == np.intersect1d(np.nonzero(d1), np.nonzero(d2)).max():
                    if cosine_similarity([d1], [d2])[0][0] >= threshold and id1 != id2:
                        res.append((id1, id2, cosine_similarity([d1], [d2])[0][0]))
        return res
    
    
    
    # Create SparkSession 
    '''spark = SparkSession.builder \
        .setMaster(f"local[{workers}]") \
    	.config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
    	.appName("all_pairs_docs_similarity.com") \
    	.getOrCreate()
    sc = spark.sparkContext'''

    conf = SparkConf().setMaster(f"local[{workers}]") \
        .setAppName("all_pairs_docs_similarity.com") \
        .set("spark.executor.memory", "5g") \
        .set("spark.driver.memory", "5g")
    sc = SparkContext(conf=conf)
    
    

    results = {}
    

    for datasets_name, docs_list in pre_processed_data.items():
        
        print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')
        
        tfidf_features = 0
        list_pre_rrd = []
        docs_list = sample_dict(docs_list)

        # Create the features and columns vectors and list of key value pairs
        vectorizer = TfidfVectorizer()
        
        
        tfidf_features = vectorizer.fit_transform(list(docs_list.values()))
        list_pre_rrd = list(zip(list(docs_list.keys()), tfidf_features.toarray()))
        d_star = np.max(tfidf_features.toarray().T, axis=1)
        

  
        print('\nComputing b_d')
        b_d = parallel_b_d(list_pre_rrd, d_star)
        print(' DONE')
        
        
        print('\nRDD creation...')
        rdd = sc.parallelize(list_pre_rrd, numSlices=1000)
        #rdd = sc.parallelize(list_pre_rrd, numSlices=considered_docs)
        print(' DONE')

        
        print('\nAdding flatMap (map_fun) transformation...')
        mapped = rdd.flatMap(map_fun)
        print(' DONE')
        #print('\nDebug Print of the first mapped value')
        #print(mapped.first())
        
        #mapped = list(map(map_fun, list_pre_rrd1))
        #print(mapped)


        print('\nAdding groupByKey transformation...')
        grouppedby = mapped.groupByKey().mapValues(list)
        print(' DONE')
        #print('\nDebug Print of the first grouppedby value')
        #print(grouppedby.first())
        
        #reduced = mapped.reduceByKey(lambda key, val: reduce_fun(key, val))
        print('\nAdding flatMap (reduce_fun) transformation...')
        reduced = grouppedby.flatMap(reduce_fun)
        print(' DONE')
        
        
        reduced.cache()
        
        print('\nRunning .collect() action with all transformations')
        start = time.time()
        reduced_results = reduced.collect()
        end = time.time()
        print(' DONE')
        
        print('\nSimilar Documents: ')
        for tuple in reduced_results: print(tuple)

        results[datasets_name] = {'threshold': threshold, 'uniqie_pairs_sim_docs': len(reduced_results) // 2, 'elapsed': end-start}

    sc.stop()

    return results