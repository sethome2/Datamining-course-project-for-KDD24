# GCCAD

**Source repo:** https://github.com/THUDM/GraphCAD/

please run pip install -r requirements.txt to install the required packages, we used more packages than the original repo.

## Running Steps

``` 
python encoding.py --path your_pub_dir --save_path embedding_save_path

python build_graph.py --author_dir train_author_path --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python build_graph.py --author_dir ./dataset/ind_valid_author.json --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python train.py --train_dir ./dataset/train.pkl --test_dir ./dataset/valid.pkl --is_global False

python encoding.py --save_path ./dataset/roberta_embeddings_modified.pkl
python build_graph.py --author_dir ./dataset/train_author.json --save_dir ./dataset/train1.pkl --save_data_dir dataset/pid_to_info_all_clean.json --embeddings_dir ./dataset/roberta_embeddings_modified.pkl
python build_graph.py --author_dir ./dataset/ind_valid_author.json --save_dir ./dataset/valid1.pkl --save_data_dir dataset/pid_to_info_all_clean1.json --embeddings_dir ./dataset/roberta_embeddings_modified.pkl
python train.py --train_dir ./dataset/train1.pkl --test_dir ./dataset/valid1.pkl --pooling attention
```

- build_graph.py and encoding.py in GCN and GCCAD are the same