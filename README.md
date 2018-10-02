<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;"># RedMAssessment</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;"># Special thanks to <span style="margin: 0px;"><a href="https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24">Topic Modeling and Latent Dirichlet Allocation (LDA) in Python</a></span></span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">&nbsp;</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Main Input File: topic_modelling.ipynb, topic_modeling_data.json</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Output file: topic_document_lda_model.json</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="font-size: 12pt;"><strong><span style="margin: 0px; line-height: 106%;">Methodology:</span></strong></span></p>
<ul style="list-style-type: disc;">
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Import the required libraries for initial analysis</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Load JSON Data provided</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Convert the json object to dataframe for easy usage</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Data Preprocessing using 'genism' and 'nltk' libraries&shy;&shy;&shy;&shy;&shy;&shy;&shy;</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Finding the Bag of words on the dataset</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">TF-IDF of the words</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Running LDA using Bag of Words </span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Running LDA using TF-IDF corpus</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Visualising the models using the 'pyLDAvis' library</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Classification of the topics</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Performance evaluation by classifying sample document using LDA Bag of Words model</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Performance evaluation by classifying sample document using LDA TF-IDF model</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Testing model on seen document using one of the documents</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Updating the topic and probability columns with respective values in decreasing order</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Dropping the unnecessary columns from the dataframe documents</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Creating a list of dictionaries in a proper format</span></li>
<li style="margin: 0px 0px 10.66px 24px;"><span style="margin: 0px; line-height: 106%; font-size: 12pt;">Saving the json file</span></li>
</ul>
<p>The following libraries were used in this exercise</p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">pandas</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">json</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">numpy</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">gensim <span style="margin: 0px;">&nbsp; </span># for NLP<span style="margin: 0px;">&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">nltk<span style="margin: 0px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span style="margin: 0px;">&nbsp; </span># for NLP</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">pickle<span style="margin: 0px;">&nbsp;&nbsp; </span><span style="margin: 0px;">&nbsp; </span># for saving the models for future use</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">pyLDAvis<span style="margin: 0px;">&nbsp; </span># for visualization</span></p>
<p style="margin: 0px 0px 10.66px;"><span style="margin: 0px; line-height: 107%; font-size: 12pt;">&nbsp;</span></p>
<h4 style="margin: 0px 0px 10.66px;"><span style="font-size: 12pt;"><strong><span style="margin: 0px; line-height: 107%;">Instructions</span></strong></span></h4>
<p><span style="background-color: transparent; color: #000000; font-family: Verdana,Arial,Helvetica,sans-serif; font-size: 16px; font-style: normal; font-variant: normal; font-weight: 400; letter-spacing: normal; line-height: 17.33px; orphans: 2; text-align: left; text-decoration: none; text-indent: 0px; text-transform: none; -webkit-text-stroke-width: 0px; white-space: normal; word-spacing: 0px; margin: 0px;">topic_modelling.ipynb</span> file and&nbsp;<span style="background-color: transparent; color: #000000; font-family: Verdana,Arial,Helvetica,sans-serif; font-size: 16px; font-style: normal; font-variant: normal; font-weight: 400; letter-spacing: normal; line-height: 17.33px; orphans: 2; text-align: left; text-decoration: none; text-indent: 0px; text-transform: none; -webkit-text-stroke-width: 0px; white-space: normal; word-spacing: 0px; margin: 0px;">topic_modeling_data.json</span> file must be present in the same directory.</p>
<p>Open the python notebook&nbsp;<span style="background-color: transparent; color: #000000; font-family: Verdana,Arial,Helvetica,sans-serif; font-size: 16px; font-style: normal; font-variant: normal; font-weight: 400; letter-spacing: normal; line-height: 17.33px; orphans: 2; text-align: left; text-decoration: none; text-indent: 0px; text-transform: none; -webkit-text-stroke-width: 0px; white-space: normal; word-spacing: 0px; margin: 0px;">topic_modelling.ipynb and run all the cells to get the desired output.</span></p>
<p>The TDF model is built using the bag of words corpus.&nbsp;</p>
<p>The topics are classified using the same model.</p>
<p><span style="display: inline !important; float: none; background-color: transparent; color: #000000; cursor: text; font-family: Verdana,Arial,Helvetica,sans-serif; font-size: 14px; font-style: normal; font-variant: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: left; text-decoration: none; text-indent: 0px; text-transform: none; -webkit-text-stroke-width: 0px; white-space: normal; word-spacing: 0px;">Visualizations are constructed using the pyLDAvis library in this notebook.</span></p>
<p>The dictionary, corpus, models and graphs are visualized and stored in their respective formats. If these files are not necessary, please comment out the corresponding codes.</p>
<p>topic_document_lda_model.json is the final deliverable required for further processing.</p>
<p style="margin: 0px 0px 10.66px;"><span style="font-family: Calibri;">&shy;&shy;&shy;</span></p>
