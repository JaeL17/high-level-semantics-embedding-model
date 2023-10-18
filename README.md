# Overview
Welcome to the repository for high-level semantic embedding model for intent classification, which is a common task in Natural Language Processing. In this project, I have developed a high-level semantic embedding model, which focuses on a comprehensive understadning of the intention of sentences.

# Motivation
Intent classification is a task that requires a broader understanding of the intention of a sentence. However, conventional embedding models tend to  focus on specific details within a sentence, such as certain keywords. To address this limitation, I implement a data augmentation method using entity attribute and entity relation information from the FewRel relation classification dataset. You can access the FewRel dataset here: https://paperswithcode.com/dataset/fewrel

# Contents
1. **Data Parsing**: Code to construct training dataset.
2. **Training**: Code to fine-tune open-source sentence embedding models on the training dataset.
3. **Testing**: Code to compute the top-3 accuracy of **MTEB/mtop_intent** dataset using a retrieval task approach based on Annoy index.
4. **Visualising Attention**: Code to visualise attention of each layer of the embedding models.

# Running the code
1. **Data Parsing**
```
python data/data_parser.py
```

2. **Training**
#### Single GPU
```
nohup python -u train.py --base_model "sentence-transformers/all-MiniLM-L12-v2" --triplet_margin 0.5 --train_batch_size 32 --output_model_name trained_model >> logs/train_log.txt &
tail -f logs/train_log.txt
```
#### Multi-GPUs
```
nohup python -u train_multi_gpus.py --base_model "sentence-transformers/all-MiniLM-L12-v2" --triplet_margin 0.5 --train_batch_size 128 --output_model_name trained_model >> logs/train_multi_gpus_log.txt &
tail -f logs/train_multi_gpus_log.txt
```

3. **Testing**
```
test
nohup python -u test.py --test_model_path "trained_model" --annoy_tree_num 20 >> logs/test_log.txt &
tail -f logs/test_log.txt
```
## Test Results and Performance Comparison
|Model|Top-1|Top-2|Top-3|
|---|---|---|---|
|Base|0.851|0.918|0.943|
|high-level-semantic|**0.930**|**0.954**|**0.963**|
|E5-Large|0.888|0.940|0.956|
|Ember-v1|0.902|0.945|**0.963**|
