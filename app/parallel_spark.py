from pyspark.sql import SparkSession
from utils import threshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Map functuion
def map_fun(pair):
    doc_id, tf_idf_list = pair
    return [(idx, (doc_id, tf_idf_list)) for idx in range(len(tf_idf_list))]

# Reduce function
def reduce_fun(doc_id_doc_list):
    res = []
    for id1, d1 in doc_id_doc_list:
        res.extend(
            (id1, id2, cosine_similarity(d1, d2))
            for id2, d2 in doc_id_doc_list
            if cosine_similarity(d1, d2) >= threshold
        )
    return res


def pyspark_APDS(pre_processed_data):
	# Create SparkSession 
	spark = SparkSession.builder \
		.master('local[1]') \
		.config("spark.driver.memory", "15g") \
		.appName("all_pairs_docs_similarity.com") \
		.getOrCreate()

	sc = spark.sparkContext
 
	results = {}

	for datasets_name, docs_list in pre_processed_data.items():
		print(f'\nPyspark All Documents Pairs Similarities - {datasets_name}')
  
		# Create the features and columns vectors
		vectorizer = TfidfVectorizer()
		tfidf_features = vectorizer.fit_transform(list(docs_list.values()))
		#tfidf_columns = vectorizer.get_feature_names_out()
  
		dict_pre_rrd = list(
			zip(docs_list.keys(), tfidf_features.toarray())
		)


		rdd = sc.parallelize(dict_pre_rrd)
  
		print('\nDebug Print of the first 5 rdd values')
		for val in rdd.take(5): print(val)
  
		'''l = [
			(0, ('MED-10', np.random.rand(6))),                           
			(1, ('MED-10', np.random.rand(6))),
			(2, ('MED-10', np.random.rand(6))),
			(3, ('MED-10', np.random.rand(6))),
			(4, ('MED-10', np.random.rand(6))),
			(5, ('MED-10', np.random.rand(6))),
   
   			(0, ('MED-11', np.random.rand(6))),                           
			(1, ('MED-11', np.random.rand(6))),
			(2, ('MED-11', np.random.rand(6))),
			(3, ('MED-11', np.random.rand(6))),
			(4, ('MED-11', np.random.rand(6))),
			(5, ('MED-11', np.random.rand(6))), 
   
   			(0, ('MED-12', np.random.rand(6))),                           
			(1, ('MED-12', np.random.rand(6))),
			(2, ('MED-12', np.random.rand(6))),
			(3, ('MED-12', np.random.rand(6))),
			(4, ('MED-12', np.random.rand(6))),
			(5, ('MED-12', np.random.rand(6))), 
   
   			(0, ('MED-13', np.random.rand(6))),                           
			(1, ('MED-13', np.random.rand(6))),
			(2, ('MED-13', np.random.rand(6))),
			(3, ('MED-13', np.random.rand(6))),
			(4, ('MED-13', np.random.rand(6))),
			(5, ('MED-13', np.random.rand(6))), 
   
   			(0, ('MED-14', np.random.rand(6))),                           
			(1, ('MED-14', np.random.rand(6))),
			(2, ('MED-14', np.random.rand(6))),
			(3, ('MED-14', np.random.rand(6))),
			(4, ('MED-14', np.random.rand(6))),
			(5, ('MED-14', np.random.rand(6))),                   
			
		]'''
  
  
		mapped = rdd.flatMap(map_fun)
		print('\nDebug Print of the first 5 mapped values')
		for val in mapped.take(5): print(val)

		grouppedby = mapped.groupByKey().mapValues(list)
		print('\nDebug Print of the first 5 grouppedby values')
		for val in grouppedby.take(5): print(val)
  
		reduced = grouppedby.reduceByKey(reduce_fun)
		print('\nDebug Print of the first 5 reduced values')
		for val in reduced.take(5): print(val)

		results[datasets_name] = reduced.collect()
	
	sc.stop()

	return results