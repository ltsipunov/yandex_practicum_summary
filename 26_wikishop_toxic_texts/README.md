#  WikiShop toxic texts
Project for Yandex practicum course 26

## Task
- To build a model to detect toxity of comments
- Training dataset contains 150_000 text samples
- F1 metric 0.75 required 

## Tools
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- nltk
- spacy
- catboost

## Files
- wikishop.ipynb - main project file
- roberta_pretrained.ipynb - demonstrates  pretrained SkolkovoInstitute/roberta_toxicity_classifier

## Results
- TF-IDF model trained
- Found optimal vocabulary size 
- F1 metric 0.78 achieved in tests
- Found key features which the model uses to detect toxicty   

## TODO
- Roberta model, pretrained on toxic vocabulary, detects toxic texts with metric F1=0.83,  
- so the result may be improved with neural networks 
- build BERT model with performance comparable with pretrained Roberta model  

 
