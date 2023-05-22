from utils import download_dataset, documents_preprocessing
from sequential import squential_APDS
from parallel_spark import pyspark_APDS
import os

if __name__ == "__main__":
	datasets = ['nfcorpus'] # Choosen datasets
	
	datasets_data = {dataset: download_dataset(dataset) for dataset in datasets}
    
	pre_processed_data = {dataset: documents_preprocessing(dataset, docs_dict) for dataset, docs_dict in datasets_data.items()}

	result_classic, result_np = squential_APDS(pre_processed_data) 
	print('\nClassic Sequential Version Results: ', result_classic)
	print('\nNumpy Sequential Version Results: ', result_np)

	print('\n\n')
	print('-------------------- PySpark --------------------')
	pyspark_results = pyspark_APDS(pre_processed_data)#, workers=8)
	print('\nPySpark Parallel Version Results: ', pyspark_results) 
 
	#os.system('../../spark-3.4.0-bin-hadoop3/sbin/stop-worker.sh')
	#os.system('../../spark-3.4.0-bin-hadoop3/sbin/stop-master.sh')
 
 
