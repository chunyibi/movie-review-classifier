# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:59:02 2019

@author: chuny
"""

# import useful python packages
import requests
import time      
from bs4 import BeautifulSoup
import pandas as pd

# pick movies need to be scarped and set the number of pages to parse
movies=["gangs_of_new_york","parasite_2019","the_lighthouse_2019","harry_potter_and_the_goblet_of_fire"
        ,"batman_begins","the_hunger_games","the_amazing_spider_man","maleficent_2014","wolverine","transformers_revenge_of_the_fallen"]
pageNum=input("Input the number of review pages to parse: ")

# append all necessary data into a list
data=[]
my_headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
for movie in movies:
    for p in range(1,int(pageNum)+1):
        page="https://rottentomatoes.com/m/"+movie+"/reviews?page="+str(p)
        scr=False
    
        for i in range(1,6):
            try:
                response=requests.get(page, headers=my_headers)
                scr=response.content
                break
            except:
                print("Failed attempt #",i)
                time.sleep(2)
        if not scr:
            print("Could not get page:",page)
            continue
        else:
            print("Successfully got page:",page)
    
        soup = BeautifulSoup(scr.decode('ascii', 'ignore'), 'lxml')
        totals=soup.find_all("div",{"class":"row review_table_row"})
        for total in totals:
            name= "NA"
            source= "NA"
            rate  = "NA"
            review = "NA"
            date= "NA"
        
            n=total.find("a",attrs={"class":"unstyled bold articleLink"})
            if n:
                name=n.text.strip()
            s=total.find("em",attrs={"class":"subtle critic-publication"})
            if s:
                source=s.text.strip()
            r=total.find("div",attrs={'class': 'review_icon'})
            if r:
                rate=r.attrs["class"][3]
            d=total.find("div",attrs={'class': 'review-date'})
            if d:
                date=d.text.strip()
            rev=total.find("div",attrs={'class': 'the_review'}).text.strip()
            if rev:
                review=rev
        
            data.append([movie,name,rate,source,review,date])

# put review data into a pandas data frame         
headers=["Movie Name","Reviewer Name","Rating","Source","Review Text","Date"] 
df=pd.DataFrame(data,columns=headers)
df.to_excel("chunyi_bi_movies_parser.xlsx")

df.info()


#-----------------------------------------------------------------------------------------------------
# Model One: use vectorizers to learn what tokens exist in the text data and build regression model

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# binarize the "Rating" data to Rating_fresh
for y in df["Rating"].unique()[0:-1]:
    df["Rating" + "_" + y] = pd.Series(df["Rating"] == y, dtype=int)
    # Drop original feature
    df = df.drop(["Rating"], 1)
    
X_text=df["Review Text"]
Y=df["Rating_fresh"]

# create a vectorizer that will track text as binary features and build regression model 
binary_vectorizer=CountVectorizer(binary=True)
binary_vectorizer.fit(X_text)
Xb=binary_vectorizer.transform(X_text)

logistic_regression=LogisticRegression()
accs_model1 = cross_val_score(logistic_regression, Xb, Y, scoring="accuracy", cv=5)
print("Accuracy of classifier one is " + str(round(np.mean(accs_model1), 3)))

# check the base rate
accuracy=Y.mean()
#-----------------------------------------------------------------------------------------------------
# Model Two: Use TF-IDF to build the regression model

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_text)
Xt = tfidf_vectorizer.transform(X_text)

logistic_regression = LogisticRegression()
accs_model2 = cross_val_score(logistic_regression, Xt, Y, scoring="accuracy", cv=5)
print("Accuracy of classifier two is " + str(round(np.mean(accs_model2), 3)))

#-----------------------------------------------------------------------------------------------------
# Model Three: use sentimentIntensityAnalyzer and DecisionTree to analysis the review text

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
 
# use Sentiment Intensity Analyzer to get polarity scores for each review text
reviews=df["Review Text"]
sentiment=[]
s=SentimentIntensityAnalyzer()
for x in reviews:
    score=s.polarity_scores(x)
    sentiment.append(score)
df_s=pd.DataFrame(sentiment)

# add palraity scores into original data frame
df["pos_score"]=df_s["pos"]
df["neu_score"]=df_s["neu"]
df["neg_score"]=df_s["neg"]
df["comp_score"]=df_s["compound"]


predictors=["pos_score","neu_score","neg_score","comp_score"]
target="Rating_fresh"
cleaned_df = df.dropna()

# build the regression classifier
logistic_regression=LogisticRegression()
accs_model3 = cross_val_score(logistic_regression,cleaned_df[predictors],cleaned_df[target], scoring="accuracy", cv=5)
print("Accuracy of classifier three is " + str(round(np.mean(accs_model3), 3)))

# build the decision tree classifier
decision_tree=DecisionTreeClassifier(max_depth=3, criterion="entropy")
decision_tree.fit(cleaned_df[predictors],cleaned_df[target])

Y_predicted=decision_tree.predict(cleaned_df[predictors])
accs_model4=accuracy_score(Y_predicted,cleaned_df[target])
print("Accuracy of classifier four is"+str(accs_model4))

# check the learning curve of this decision tree model
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

training_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
accuracies = []

for training_percentage in training_percentages:
    X_train, X_test, Y_train, Y_test = train_test_split(cleaned_df[predictors], cleaned_df[target], train_size=training_percentage)

    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(X_train, Y_train)
    Y_test_predicted = tree.predict(X_test)
    acc = accuracy_score(Y_test_predicted, Y_test)
    accuracies.append(acc)

plt.plot(training_percentages, accuracies)
plt.show()

#-----------------------------------------------------------------------------------------------------
# Visualization
 
# Visualizing Decision Tree
from sklearn import tree
import pydotplus
from IPython.display import Image  

dot_data = tree.export_graphviz(decision_tree,
                                out_file=None,
                                feature_names=predictors,  
                                class_names=["n", "y"])

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
graph.write_png("tree.png")

# Histogram shows fresh/rotten reviews counts for each movie
import seaborn as sns
import matplotlib.pyplot as plt

x=df["Movie Name"].value_counts().reset_index()
y=df.groupby("Movie Name").sum().reset_index()
y["Total Count"]=x["Movie Name"]
y_sort=y.sort_values(by="Rating_fresh",ascending=False)

sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 25))
sns.set_color_codes("pastel")
sns.barplot(x="Total Count", y="Movie Name", data=y_sort,
            label="Total", color="b")
sns.set_color_codes("muted")
sns_plot=sns.barplot(x="Rating_fresh", y="Movie Name", data=y_sort,
            label="Fresh", color="b")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 200), ylabel="Moview",
       xlabel="Fresh ratings over Total ratings")
sns.despine(left=True, bottom=True)
fig = sns_plot.get_figure()
fig.savefig("output.png")

# use scatter plot to discover the relationship between time and count of fresh ratings for specific movie
import matplotlib

# for movie gangs_of_new_york
m=df[df["Movie Name"] == "gangs_of_new_york"]
year_list=[]
for d in m["Date"]:
    year=d[-4:]
    year_list.append(year)
m["year"]=year_list
count_fresh=m.groupby("year").sum().reset_index()
count_rating=m.groupby("year").count().reset_index()
count_rating["count_fresh"]=count_fresh["Rating_fresh"]
count_rating["percentage"]=count_rating["count_fresh"]/count_rating["Rating_fresh"]

matplotlib.pyplot.scatter(count_rating["year"], count_rating["percentage"])
plt.savefig('fig1.png')

# for movie the_hunger_games
m=df[df["Movie Name"] == "the_hunger_games"]
year_list=[]
for d in m["Date"]:
    year=d[-4:]
    year_list.append(year)
m["year"]=year_list
count_fresh=m.groupby("year").sum().reset_index()
count_rating=m.groupby("year").count().reset_index()
count_rating["count_fresh"]=count_fresh["Rating_fresh"]
count_rating["percentage"]=count_rating["count_fresh"]/count_rating["Rating_fresh"]

matplotlib.pyplot.scatter(count_rating["year"], count_rating["percentage"])
plt.savefig('fig2.png')

# for movie harry_potter_and_the_goblet_of_fire
m=df[df["Movie Name"] == "harry_potter_and_the_goblet_of_fire"]
year_list=[]
for d in m["Date"]:
    year=d[-4:]
    year_list.append(year)
m["year"]=year_list
count_fresh=m.groupby("year").sum().reset_index()
count_rating=m.groupby("year").count().reset_index()
count_rating["count_fresh"]=count_fresh["Rating_fresh"]
count_rating["percentage"]=count_rating["count_fresh"]/count_rating["Rating_fresh"]

matplotlib.pyplot.scatter(count_rating["year"], count_rating["percentage"])
plt.savefig('fig3.png')

# for movie transformers_revenge_of_the_fallen
m=df[df["Movie Name"] == "transformers_revenge_of_the_fallen"]
year_list=[]
for d in m["Date"]:
    year=d[-4:]
    year_list.append(year)
m["year"]=year_list
count_fresh=m.groupby("year").sum().reset_index()
count_rating=m.groupby("year").count().reset_index()
count_rating["count_fresh"]=count_fresh["Rating_fresh"]
count_rating["percentage"]=count_rating["count_fresh"]/count_rating["Rating_fresh"]

matplotlib.pyplot.scatter(count_rating["year"], count_rating["percentage"])
plt.savefig('fig4.png')

