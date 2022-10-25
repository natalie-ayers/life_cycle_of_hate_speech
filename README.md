# Life Cycle of Hate Speech

[Project Report](https://github.com/natalie-ayers/life_cycle_of_hate_speech/blob/main/Adv%20ML%20Final%20Paper.pdf)

## Project Description

We hope to examine the progression and varying expression of hate speech through three lenses: search platforms, social media posts, and occurring events. 


## Team Members
- Natalie Ayers
- Sophia Mlawer
- Yangzhou Ou


## Repository structure
### Data Collection and Preprocessing
- `data/`: orignal tweet datasets as well as cleaned datasets and upsampled ones
- `collect_twitter_data.ipynb`: scrape tweet info with snscrape package; 140 lines
- `clean_data.ipynb`: data preprocessing (including text normalization, tokenization, etc.); 171 lines
- `exploratory_data_analysis.ipynb `: training data content evaluation: 53 lines

### Google Trends
- `Hate Speech.ipynb`: get Google trends data for different types of hate speech: 65 lines

### Model Training
- `log_regression.ipynb`: Create train/val/test splits; create vocab and basic features from training data; feature importance pipeline; log regression training and evaluation: 517 lines
- `BoW_XGBoost_NB_SVC_LR_GB_RF_KNN.ipynb`: Bag-of-Words-based models training and evaluation pipeline: 171 lines
- `TFIDF_XGBoost_NB_SVC_LR_GB_RF_KNN.ipynb`: TF-IDF-based models training and evaluation pipeline: 532 lines
- `LSTM.ipynb`: bi-LSTM model training and evaluation pipeline: 714 lines
- `transformers_bert.ipynb`: distilBERT model training and evaluation pipeline: 415 lines
- `CNN.ipynb`: CNN models training and evaluation pipeline: 739 lines
- `model.pt`: best-performing LSTM model
- `classify_with_trained_models.ipynb`: utilize `model.pt` to predict hate speech in Waseem and Hovy twitter corpus; compare predictions with Google Trends patterns: 243 lines
