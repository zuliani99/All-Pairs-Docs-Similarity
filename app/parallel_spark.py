from pyspark.sql import SparkSession
from utils import threshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def pyspark_APDS(pre_processed_data):
    # Create SparkSession 
    spark = SparkSession.builder \
    	.master('local[*]') \
    	.config("spark.driver.memory", "10g") \
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
                if term == np.intersect1d(d1, d2).max():
                    if cosine_similarity([d1], [d2])[0][0] >= threshold and id1 != id2:
                        res.append((id1, id2, cosine_similarity([d1], [d2])[0][0]))
        return res
    

    for datasets_name, docs_list in pre_processed_data.items():
        
        print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')

        # Create the features and columns vectors
        vectorizer = TfidfVectorizer()
        tfidf_features = vectorizer.fit_transform(list(docs_list.values())[:considered_docs])
        
        dict_pre_rrd = list(
        	zip(list(docs_list.keys())[:considered_docs], tfidf_features.toarray())
        )
        
        d_star = np.max(tfidf_features.toarray().T, axis=1)
        
        b_d = {}
        for doc_id, doc_tfidf in dict_pre_rrd:
            temp_product_sum = 0  
            sorted_indices = np.argsort(doc_tfidf)
            sorted_tfidf = sorted(doc_tfidf)  
            for termid, tfidf in zip(sorted_indices, sorted_tfidf):
                temp_product_sum += tfidf * d_star[termid]
                if temp_product_sum >= threshold:
                    b_d[doc_id] = termid - 1
                    break

        rdd = sc.parallelize(dict_pre_rrd)

        print('\nDebug Print of the first rdd value')
        print(rdd.first())

        mapped = rdd.flatMap(map_fun)
        print('\nDebug Print of the first mapped value')
        print(mapped.first())

        grouppedby = mapped.groupByKey().mapValues(list)
        print('\nDebug Print of the first grouppedby value')
        print(grouppedby.first())
        
        #reduced = mapped.reduceByKey(reduce_fun)
        reduced = grouppedby.flatMap(reduce_fun)
        print('\nDebug Print of the first reduced values')
        reduced_results = reduced.collect()
        print(reduced_results)
        
        reduced.cache()

        results[datasets_name] = reduced_results

    sc.stop()

    return results