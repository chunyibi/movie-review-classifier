# Movie-Review-Classifier
The purpose of this project is to do basic text analysis for movie reviews, to build classifier that can classify review text as negative or positive, and to provide interesting insights of data using visualization tools.
## Method approach
<ol>
<li> Data collection: Scraping data from the website and putting data into a pandas data frame;
<li> Model building: Using sklearn CountVectorizer package, TF-IDF package and Sentiment Intensity Analyzer package to do text analysis; using regression and decision tree to build models and evaluating performance by models’ accuracy of prediction;
<li> Data visualization: Visualizing models and interesting insights of data.
</ol>

## Data Collection
<b> Python packages:</b>  request, time, beautifulsoup , pandas<br>
<b>Data Source: </b> scraped 10 pages of movie review from 10 randomly selected movies (Gangs of New York, Parasite, the Lighthouse, Harry Potter and the Goblet of Fire, Batman Begins, the Hunger Games, the Amazing Spider Man, Maleficent, Wolverine, Transformers Revenge of the Fallen). Get necessary information for each review, format data and put those into a pandas data frame “df” with 6 columns and 2000 rows.
## Model Building
<b> Model 1: </b>
<menu>
<li type="disc"> <b>Packages:</b> numay, sklearn.feature_extraction.text, sklearn.linear_model, sklearn.model_selection
<li type="disc"> <b>Steps:</b>  1. binarized the "Rating" data;  2. created a vectorizer that will track text as binary features;  3. built regression model and evaluated its accuracy
</menu>
<br>
<b> Model 2: </b>
<menu>
<li type="disc"> <b>Packages:</b> TF-IDF, sklearn.linear_model
<li type="disc"> <b>Steps:</b>  1.used TF-IDF to vectorize the text;  2. built regression model and evaluated its accuracy
</menu>
<br>
<b> Model 3: </b>
<menu>
<li type="disc"> <b>Packages:</b> nltk.sentiment.vader, sklearn.linear_model
<li type="disc"> <b>Steps:</b>  1.used Sentiment Intensity Analyzer to get polarity scores for each review text;  2.built regression model and evaluated its accuracy
</menu>
<br>
<b> Model 4: </b>
<menu>
<li type="disc"> <b>Packages:</b> nltk.sentiment.vader, sklearn.tree
<li type="disc"> <b>Steps:</b>  1.used the same Sentiment Intensity Analyzer score in model 3;  2.built decision tree model and evaluated its accuracy
</menu>

## Visualization and Data Insights
<b> Model Visualization: </b> Pydotplus, sklearn, Ipython.display packages are used to display the decision tree structure
<br><b> Data Insights 1: </b> Seaborn Histogram shows fresh reviews counts for each movie;
<br>“Parasite” has the most fresh ratings (197 out of 200) ;
<br>“Transformers revenge of the fallen” has the least fresh ratings (33 out of 200)
<br><b> Data Insights 2: </b> there is no obvious relationships between movie reviews and time (review ratings will not be better or worse when time goes by)

## Supports:
Run the program, if you have any questions, reach out to me at one of the following places!
<ol><li><b>Email:</b> chunyibi@gmail.com
<li> <b>LinkedIn:</b> Chunyi Bi</ol>
