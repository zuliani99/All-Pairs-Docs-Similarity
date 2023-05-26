from typing import Dict, List, Tuple
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import csv

try:
	import ipywidgets
	from tqdm.auto import tqdm
except ModuleNotFoundError:
	from tqdm import tqdm
	
import pandas as pd
import spacy
import random


nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
# Lambda for text pre-processing
clean_tokens = lambda tokens : ' '.join([token.lemma_.lower() for token in tokens if token not in stopwords and not token.is_punct])
	

def download_dataset(dataset: str) -> Dict[str, List[str]]:
	'''
	PURPOSE: download the dataset
	ARGUMENTS:
		- dataset (str): string describing the beir dataset
	RETURN:
		- (Dict[str, List[str]]:) dictionary of documents
	'''
 
	data_path = f'datasets/{dataset}'
	if not os.path.isdir(data_path):
		url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
		out_dir = os.path.join(os.getcwd(), 'datasets')
		data_path = util.download_and_unzip(url, out_dir)
		print(f'Dataset downloaded here: {data_path}')
	corpus, _, _ = GenericDataLoader(data_path).load(split="test")
	return {doc_id: title_text['title'] + ' ' + title_text['text'] for doc_id, title_text in corpus.items()}


def pre_process(dictionary: Tuple[str, str]) -> Dict[str, str]:
	'''
	PURPOSE: preprocess a single document
	ARGUMENTS:
		- dictionary (Tuple[str, str]): tuple of docid and document
	RETURN:
		- (Dict[str, str]): docid and preprocesed document text
	'''
 
	key, value = dictionary
	return {key: clean_tokens(nlp(value))}



def documents_preprocessing(dataset_name: str, documents: Dict[str, str]) -> Dict[str, str]:
	'''
	PURPOSE: pre-process all set of documents od a single dataset
	ARGUMENTS:
		- dataset_name (str): dataset name
		- documents: (Tuple[str, str]): dictionary of docid and document
	RETURN:
		- (Dict[str, str]): docid and preprocesed document text
	'''
 
	path_datasets = os.path.join(os.getcwd(), 'datasets')
	if os.path.exists(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet')):
		return pd.read_parquet(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet')).to_dict()[0]
	
 
	new_documents = {}

	# Parallel execution of the pre-processing step
	with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
		results = list(
				tqdm(
					executor.map(pre_process, documents.items()),
					total=len(documents),
					desc=f'{dataset_name} - Documents Pre-Processing',
				)
			)

	for result in results:
		new_documents.update(result)

	# Write the pre-processed dictionaty of documents
	write_pd = pd.DataFrame.from_dict(new_documents, orient='index')
	write_pd.to_parquet(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet'))

	return new_documents


def sample_dict(dictionary: Dict[str, str], considered_docs: int | None) -> Dict[str, str]:
	'''
	PURPOSE: sample a set of element of the document dictionary
	ARGUMENTS:
		- dictionary (Dict[str, str]): document dictioanry
		- considered_docs: (int | None): number of considered document
	RETURN:
		- (Dict[str, str]): docid and preprocesed document text
	'''
 
	if considered_docs is None: # In case is none it means that I want the whole set
		return dictionary
	keys = list(dictionary.keys())  # Get a list of keys from the dictionary
	sampled_keys = random.sample(keys, considered_docs)  # Sample from the list of keys
	return {key: dictionary[key] for key in sampled_keys}
	

def create_doc_sim_csv(pairs_list: List[Tuple[str, str, float]], ds_name: str,
					   threshold: float, type: str | None, workers: None | int = None ) -> None: 
	'''
	PURPOSE: create the .csv file sotring the list of similar documents pairs with the cosine similarity
	ARGUMENTS:
		- pairs_list (List[Tuple[str, str, float]]): list of unique similar pair with the similarity
		- ds_name: (str): dataset name
		- threshold (float): used threshold
		- type (str | None): type of sequential version
		- workers (None | int): number of workers used
	RETURN: None
	'''

	path = ''
	if type is not None:
		if not os.path.exists(f'./results/{ds_name}/{threshold}'): os.makedirs(f'./results/{ds_name}/{threshold}')
		path = f'./results/{ds_name}/{threshold}/{type}_sequential.csv'
	else:
		if not os.path.exists(f'./results/{ds_name}/{threshold}/pyspark/'): os.makedirs(f'./results/{ds_name}/{threshold}/pyspark/')
		path = f'./results/{ds_name}/{threshold}/pyspark/{workers}_workers.csv'
	if not os.path.exists(path): # If there is already a file, return
		with open(path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerows(pairs_list)
