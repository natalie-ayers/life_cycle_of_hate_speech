# life_cycle_of_hate_speech

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
- `clean_data.ipynb`: data preprocessing (including text normalization, tokenization, etc.); 165 lines
- `exploratory_data_analysis.ipynb `: 

### Google Trends
- `Hate Speech.ipynb`: get good trends data for different types of hate speech: 65 lines

### Model Training
- `cleaning_scratchwork.ipynb`:
- `BoW_XGBoost_NB_SVC_LR_GB_RF_KNN.ipynb`: Bag-of-Words-based models training and evaluation pipeline: 171 lines
- `TFIDF_XGBoost_NB_SVC_LR_GB_RF_KNN.ipynb`: TF-IDF-based models training and evaluation pipeline: 532 lines
- `LSTM.ipynb`: 714 lines
- `sklearn_log.py`:
- `transformers_bert.ipynb`: 
- `CNN.ipynb`: CNN model training and exploration: 739 lines
- `model.pt`: best-performing LSTM model
