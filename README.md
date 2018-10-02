# RedMAssessment
# Special thanks to 
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

Main Input File: topic_modelling.ipynb, topic_modeling_data.json
Output file: topic_document_lda_model.json
Methodology:
-> Import the required libraries for initial analysis
-> Load JSON Data provided
-> Convert the json object to dataframe for easy usage
-> Data Preprocessing using 'genism' and 'nltk' libraries
-> Finding the Bag of words on the dataset
-> TF-IDF
-> Running LDA using Bag of Words 
-> Running LDA using TF-IDF corpus
-> Visualising the models using the 'pyLDAvis' library
-> Classification of the topics
-> Performance evaluation by classifying sample document using LDA Bag of Words model
-> Performance evaluation by classifying sample document using LDA TF-IDF model
-> Testing model on seen document using one of the documents
-> Updating the topic and probability columns with respective values in decreasing order
-> Dropping the unnecessary columns from the dataframe documents
-> Creating a list of dictionaries in a proper format
-> Saving the json file

topic_modelling.ipynb is the main file in this project
