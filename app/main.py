from utils import download_dataset, documents_preprocessing, sample_dict, create_doc_sim_csv
from sequential import classic_squential_APDS, numpy_squential_APDS
from parallel_spark import pyspark_APDS
import pandas as pd


if __name__ == "__main__":
	datasets = ['nfcorpus'] # Choosen datasets
	thresholds = [0.5, 0.7, 0.8, 0.9] # Choosen thresholds
	max_workers = 10 # Maximum amount of workers
	considered_docs = 1000 # Number of considere documents

	# Download datasets
	datasets_data = {dataset: download_dataset(dataset) for dataset in datasets}

	# Pre-process and sample with the original datasets
	pre_processed_data = {dataset: sample_dict(documents_preprocessing(dataset, docs_dict), considered_docs) 
	for dataset, docs_dict in datasets_data.items()}

	sequential_results = []
	pyspark_results = []

	for ds_name, sampled_dict in pre_processed_data.items():

		print(f'\n------------ALL PAIRS DOCUMENTS SIMILARITY - {ds_name}------------')

		for threshold in thresholds:
			print(f'\n--------Running with threshold: {threshold}--------')

   			# Classic & Numpy - Sequential Execution

			print('\nClassic Sequential Execution')
			sim_doc_cl, cl_res = classic_squential_APDS(ds_name=ds_name, sampled_dict=sampled_dict, threshold=threshold)
			sequential_results.append(cl_res)
			create_doc_sim_csv(sim_doc_cl, ds_name, threshold, 'classic')
			print(' Done')		

			print('\nNumpy Sequential Execution')
			sim_doc_np, np_res = numpy_squential_APDS(ds_name=ds_name, sampled_dict=sampled_dict, threshold=threshold)
			sequential_results.append(np_res)
			create_doc_sim_csv(sim_doc_np, ds_name, threshold, 'numpy')
			print(' Done')

			for workers in range(1, max_workers + 1):

				# PySpark Execution

				print(f'\nPySpark Parallel Execution with {workers} workers')
				sim_doc_ps, ps_res = pyspark_APDS(ds_name=ds_name, sampled_dict=sampled_dict, threshold=threshold, workers=workers)
				pyspark_results.append(ps_res)
				create_doc_sim_csv(sim_doc_ps, ds_name, threshold, None, workers)
				print(' Done')

			print('\n')

	print('\nSaving sequential_results')
	pd.DataFrame.from_dict(
		dict(zip(range(len(sequential_results)), sequential_results)),
		orient='index',
		columns=[
			'type',
			'ds_name',
			'elapsed',
			'threshold',
			'uniqie_pairs_sim_docs',
		],
	).to_csv('./results/sequential_results.csv', index=False)
	print(' Done')

	print('\nSaving pyspark_results')
	pd.DataFrame.from_dict(
		dict(zip(range(len(pyspark_results)), pyspark_results)),
		orient='index',
		columns=[
			'ds_name',
			'elapsed',
			'threshold',
			'uniqie_pairs_sim_docs',
   			'workers',
		],
	).to_csv('./results/pyspark_results.csv', index=False)
	print( 'Done')
 
 
