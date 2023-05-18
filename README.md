# All-Pairs-Docs-Similarity
Given a set of documents and the minimum required similarity threshold find the number of document pairs that exceed the threshold

## Requisites
```console
pip install beir
pip install pandas
pip install sklearn
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
pip install ipywidgets
```

## Used Daataset
!(nfcorpus)[https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/]


## Start Application
Simulate a cluster on a single machine:

```console
./spark-3.4.0-bin-hadoop3/sbin/start-master.sh
```

After launching the master node as above, its web user interface is available at *http://localhost:8080/* Then you can launch as many workers using as the spark://HOST:PORT the main url reported on the web UI ( above the Alive Workers entry)

```console
export SPARK_WORKER_INSTANCES=X; ./spark-3.4.0-bin-hadoop3/sbin/start-worker.sh spark://HOST:PORT
```
With `X` the number of worker you would like to run the benchmark 

Now you can launch a script on the cluster you just created, and monitor its advance on the master web UI:
```console
./spark-3.4.0-bin-hadoop3/bin/spark-submit --master spark://HOST:PORT
.app/main.py
```

## Results