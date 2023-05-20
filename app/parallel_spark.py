from pyspark.sql import SparkSession
from utils import threshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def pyspark_APDS(pre_processed_data):
    
    # Create SparkSession 
    spark = SparkSession.builder \
        .master("local[*]") \
    	.config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
    	.appName("all_pairs_docs_similarity.com") \
    	.getOrCreate()
    
    sc = spark.sparkContext
    

    results = {}
    
    considered_docs = 50
    
    # Map functuion
    def map_fun(pair):
        docid, tf_idf_list = pair
        res = []
        sorted_indices = np.argsort(tf_idf_list)
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
                if len(np.intersect1d(d1, d2)) > 0 and term == np.intersect1d(d1, d2).max():
                    if cosine_similarity([d1], [d2])[0][0] >= threshold and id1 != id2:
                        res.append([id1, id2, cosine_similarity([d1], [d2])[0][0]])
        return res
    

    for datasets_name, docs_list in pre_processed_data.items():
        
        print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')

        # Create the features and columns vectors
        vectorizer = TfidfVectorizer()
        tfidf_features = vectorizer.fit_transform(list(docs_list.values())[:considered_docs]) #[:considered_docs]
        
        dict_pre_rrd1 = list(
        	zip(list(docs_list.keys())[:considered_docs], tfidf_features.toarray()) #[:considered_docs]
        )
        
        dict_pre_rrd2 = [
            ('ciao', np.random.rand(6)),
            ('MARIO', np.random.rand(6)),
            ('CAIO', np.random.rand(6))
        ]
        
        
        d_star = np.max(tfidf_features.toarray().T, axis=1)
        #print('d_star', d_star)
        
        b_d = {}
        for doc_id, doc_tfidf in dict_pre_rrd1:
            temp_product_sum = 0  
            sorted_indices = np.argsort(doc_tfidf)
            for termid in sorted_indices:
                temp_product_sum += doc_tfidf[termid] * d_star[termid]
                if temp_product_sum >= threshold:
                    b_d[doc_id] = termid - 1
                    break
        print('b_d', b_d)
        
        print('\nRDD creation...')
        rdd = sc.parallelize(dict_pre_rrd1)#, numSlices=100)
        #rdd = sc.parallelize(dict_pre_rrd2)#, numSlices=100)
        print(' DONE')

        #print('\nDebug Print of the first rdd value')
        #print(rdd.first()) 
        
        print('\nflatMap (map_fun) transformation...')
        mapped = rdd.flatMap(map_fun)
        print(' DONE')
        #print('\nDebug Print of the first mapped value')
        #print(mapped.first())
        
        #mapped = list(map(map_fun, dict_pre_rrd2))
        #print(mapped)
        #mapped = list(map(flatten_list, mapped))
        #print(mapped)
        
        print('\ngroupByKey transformation...')
        grouppedby = mapped.groupByKey().mapValues(list)
        print(' DONE')
        #print('\nDebug Print of the first grouppedby value')
        #print(grouppedby.first())
        
        #reduced = mapped.reduceByKey(lambda key, val: reduce_fun(key, val))
        print('\nflatMap (reduce_fun) transformation...')
        reduced = grouppedby.flatMap(reduce_fun)
        print(' DONE')
        
        
        reduced.cache()
        
        print('\nRunning .collect() action')
        reduced_results = reduced.collect()
        print(reduced_results)

        results[datasets_name] = reduced_results

    sc.stop()

    return results