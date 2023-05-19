from utils import download_dataset, documents_preprocessing
from sequential import squential_APDS
from parallel_spark import pyspark_APDS
import os

if __name__ == "__main__":
	datasets = ['nfcorpus'] # Choosen datasets
	
	datasets_data = {dataset: download_dataset(dataset) for dataset in datasets}
    
	pre_processed_data = {dataset: documents_preprocessing(dataset, docs_dict) for dataset, docs_dict in datasets_data.items()}

	result_classic, result_np = squential_APDS(pre_processed_data) 
	print('\nClassic Sequential Version Result: ', result_classic)
	print('\nNumpy Sequential Version Result: ', result_np)

	print('\n\n')
	pyspark_results = pyspark_APDS(pre_processed_data)
	print(pyspark_results) 
 
	#os.system('../../spark-3.4.0-bin-hadoop3/sbin/stop-worker.sh')
	#os.system('../../spark-3.4.0-bin-hadoop3/sbin/stop-master.sh')
 
 
 
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