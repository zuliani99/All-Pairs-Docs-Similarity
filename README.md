# All-Pairs-Docs-Similarity
Given a set of documents and the minimum required similarity threshold find the number of document pairs that exceed the threshold

## Requisites
```console
sudo apt install default-jre
```

```console
pip install beir
pip install pandas
pip install sklearn
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
pip install ipywidgets
```

## PySpark Local Installation
```console
wget https://dlcdn.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
sha512sum spark-3.4.0-bin-hadoop3.tgz
tar -xzf spark-3.4.0-bin-hadoop3.tgz

pip install pyspark
```

## Used Daataset
[nfcorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)


## Start Application
Start *PySpark* typing:
```console
./spark-3.4.0-bin-hadoop3/bin/pyspark
```

And memorize the location of HOST:PORT of the local server

Simulate a cluster on a single machine:

```console
./spark-3.4.0-bin-hadoop3/sbin/start-master.sh
```

After launching the master node as above, its web user interface is available at *http://localhost:8080/* Then you can launch as many workers using as the spark://HOST:PORT the main url reported on the web UI ( above the Alive Workers entry)

```console
export SPARK_WORKER_INSTANCES=4; ./spark-3.4.0-bin-hadoop3/sbin/start-worker.sh spark://riccardo-HP-EliteBook-840-G2:7077
```

Now you can launch a script on the cluster you just created, and monitor its advance on the master web UI:
```console
./spark-3.4.0-bin-hadoop3/bin/spark-submit --master spark://riccardo-HP-EliteBook-840-G2:7077 ./All-Pairs-Docs-Similarity/app/main.py
```

## Results
