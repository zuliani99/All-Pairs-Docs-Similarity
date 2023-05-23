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
    

#def download_dataset(dataset: str) -> Dict[str, List[str]]:
def download_dataset(dataset):
	'''
	PURPOSE: download the dataset
	ARGUMENTS:
		- dataset (str): string describing the beir dataset
	RETURN:
		- (List[str]) list of documents
	'''
	data_path = f'datasets/{dataset}'
	if not os.path.isdir(data_path):
		url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
		out_dir = os.path.join(os.getcwd(), 'datasets')
		data_path = util.download_and_unzip(url, out_dir)
		print(f'Dataset downloaded here: {data_path}')
	corpus, _, _ = GenericDataLoader(data_path).load(split="test")
	return {doc_id: title_text['title'] + ' ' + title_text['text'] for doc_id, title_text in corpus.items()}


def pre_process(dictionary):
	'''
	PURPOSE: preprocess the text using spaCy
	ARGUMENTS:
		- corpus (str): string of document to pre-process
	RETURN:
		- str: cleaned document
	'''
	key, value = dictionary
	return {key: clean_tokens(nlp(value))}



#def documents_preprocessing(dataset_name: str, documents: Dict[str, str]) -> Dict[str, str]:
def documents_preprocessing(dataset_name, documents):
	'''
	PURPOSE: preprocess all the documents and query for the relative dataset
	ARGUMENTS:tfidf_features
		- dataset_name (str): string describing the dataset name
		- documents (Dict[str, List[str]]): doc_id, document_text dictionary
	RETURN: 
		- new_documents (Dict[str, List[str]]): dictionary of cleaned documents
	'''
 
	path_datasets = os.path.join(os.getcwd(), 'datasets')
	if os.path.exists(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet')):
		return pd.read_parquet(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet')).to_dict()[0]
	
 
	new_documents = {}

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

	write_pd = pd.DataFrame.from_dict(new_documents, orient='index')
	write_pd.to_parquet(os.path.join(path_datasets, dataset_name, 'pre_processed_corpus.parquet'))

	return new_documents


def sample_dict(dictionary, considered_docs):
    if considered_docs is None:
        return dictionary
    keys = list(dictionary.keys())  # Get a list of keys from the dictionary
    sampled_keys = random.sample(keys, considered_docs)  # Sample from the list of keys
    return {key: dictionary[key] for key in sampled_keys}
    

def create_doc_sim_csv(pairs_list, ds_name,  threshold, type, workers=None):
    path = ''
    if type is not None:
        if not os.path.exists(f'./results/{ds_name}/{threshold}'): os.makedirs(f'./results/{ds_name}/{threshold}')
        path = f'./results/{ds_name}/{threshold}/{type}_sequential.csv'
    else:
        if not os.path.exists(f'./results/{ds_name}/{threshold}/pyspark/'): os.makedirs(f'./results/{ds_name}/{threshold}/pyspark/')
        path = f'./results/{ds_name}/{threshold}/pyspark/{workers}_workers.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pairs_list)
