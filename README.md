# TwitterSentimentAnalysis
##### Problem Statement:

The goal of this project is to perform sentiment analysis on tweets to determine whether each tweet expresses a positive, negative, or neutral sentiment. Sentiment analysis will help us understand the overall sentiment of the Twitter users towards a particular topic, brand, or event.

The project began with the careful selection of a dataset from Kaggle. This dataset was a crucial asset in advancing our analysis, as it encompassed a diverse collection of tweets accompanied by corresponding sentiment labels – positive, negative, or neutral. This richly labeled dataset served as the foundation for training, fine-tuning, and evaluating our sentiment analysis models.

The data collection process unfolded through the following steps:

###### Data Exploration: 
Once identified, the chosen dataset was promptly procured from Kaggle's repository. Before plunging into analysis, an initial exploration was imperative. This phase provided insights into the dataset's structure, the distribution of sentiment labels, and the potential challenges posed by data quality.
###### Data Preprocessing: 
The dataset, in its raw form, underwent meticulous preprocessing to refine its quality and formatting. Duplication entries were removed, and strategies were devised to address missing data. Furthermore, started exploring into text preprocessing techniques, encompassing text tokenization, stemming, and the removal of stopwords. These measures standardized the tweet text, ensuring that it was well-prepared for subsequent feature extraction.
###### Text Labeling and Categorization: 
Our dataset, sourced from Kaggle, possessed the invaluable asset of pre-labeled sentiment categories for each tweet. These labels underpinned our machine learning endeavor, endowing us with the ability to train models to prognosticate sentiment based on tweet content. Ensuring the accuracy and consistency of the labeling process was paramount in preserving the dataset's reliability.
###### Splitting into Training and Testing Sets: 
To assess the effciancy of sentiment analysis models, a division of the dataset into training and testing sets waas made. The training set was the crucible for molding and refining our models, while the testing set, for evaluating the models' capacity for accurate prediction and generalization.
###### Exploratory Data Analysis (EDA): 
This phase unveiled insights and patterns residing within the dataset. EDA provided a window into the distribution of sentiment labels, unveiled any potential class imbalances, and furnished preliminary insight into the linguistic tapestry weaving through the various sentiment categories.





Throughout our project, we followed a consistent workflow:

Data Collection: We gathered a diverse dataset of labeled tweets encompassing positive, negative, and neutral sentiments.
Data Preprocessing: We cleaned and tokenized the text, removed stopwords, and applied techniques like stemming or lemmatization to normalize the words.
Feature Extraction: We converted the text into numerical features suitable for machine learning algorithms. TF-IDF and word embeddings were commonly used techniques.
Model Training and Evaluation: We split the dataset into training and testing sets, trained the models, and evaluated their performance using metrics like accuracy, precision, recall, and F1-score.
Hyperparameter Tuning: We fine-tuned the models by adjusting hyperparameters to achieve the best possible performance.

##### Data Preprocessing:

Data preprocessing plays a pivotal role in refining Twitter data for accurate sentiment analysis. The process involves several critical phases to ensure data quality and suitability for machine learning. Here's a concise summary of the essential data preprocessing steps:

###### 1. Raw Data Cleaning:

Convert Text to Lowercase: Standardize text case to reduce vocabulary size and variation.
Remove Mentions: Exclude '@mentions' to eliminate irrelevant user references.
Eliminate Special Characters: Strip punctuation and special characters for uniformity.
Filter Out Stopwords: Discard common, non-informative words ('the,' 'is,' 'and,' etc.).
Hyperlink Removal: Omit URLs, which don't contribute to sentiment analysis.
###### 2. Tokenization:

Tokenization Breaks text into individual words (tokens) for analysis.
Tokens serve as input features for machine learning algorithms.
###### 3. Stemming:

Stemming reduces words to their base form (stem) to capture core meanings.
For instance, 'satisfying,' 'satisfaction,' and 'satisfied' all stem to 'satisfy.'
###### 4. Vectorization:

Convert Tokens to Numeric Representation using techniques like TF-IDF or word embeddings.
Numeric vectors facilitate machine learning model input.
###### 5. Dataset Splitting:

Divide Preprocessed Data into Training and Testing Sets for model evaluation.
Effective data preprocessing enhances data quality, ensures consistency, and simplifies feature extraction, enabling sentiment analysis models to learn patterns accurately and achieve robust sentiment classification.

After preprocessing the data to ensure its quality and suitability for analysis, Proceeded to fit the sentiment analysis model using the selected algorithms: **Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, and **Naive Bayes**. Each algorithm underwent a series of steps, including feature extraction, model training, and hyperparameter tuning, to optimize their performance. Here's an overview of how we fit the models and our conclusion on which algorithm is best suited for this project:

##### Logistic Regression:
Began by encoding the preprocessed tweet text into numerical features using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency). These features were then fed into the Logistic Regression model. We performed a grid search to find the optimal regularization parameter (C) through cross-validation. The resulting model was trained on the training set and evaluated on the testing set using metrics like accuracy, precision, recall, and F1-score.
##### Decision Tree Classifier:
For the Decision Tree Classifier, Used the preprocessed features as inputs and trained the model on the training set. To avoid overfitting, we explored different tree depths and minimum samples per leaf during the tuning process. The performance of the model was assessed using the same set of evaluation metrics.
##### Random Forest Classifier:
The Random Forest Classifier was trained by aggregating the predictions of multiple decision trees. Adjusted hyperparameters such as the number of trees and maximum features per split. This approach aimed to enhance the model's generalization capabilities while avoiding overfitting.
##### Naive Bayes:
The Naive Bayes model, which is well-suited for text classification, was fitted using the preprocessed features. Then, applied techniques like TF-IDF to transform the text into numerical values. The model's performance was evaluated using the same evaluation metrics as the other algorithms.



