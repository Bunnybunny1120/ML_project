
import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

####### Training and validating without authors with one sample #######

with open("train.json") as file: 
    jdata = json.load(file)

data_full = pd.DataFrame(jdata)

######## data exploration - left as comments to improve performance while running the full code
#print(data_full.shape)
#print(data_full.columns)
#print(data_full.head(5))
#print(data_full.groupby('authorName').size())

data_full['authorId'] = pd.to_numeric(data_full['authorId'])
data_full['year'] = data_full['year'].map(str)
filtered_data = data_full[data_full['authorId'].map(data_full['authorId'].value_counts()) > 1]

X = filtered_data['abstract'] + ". " + filtered_data['title'] + ". " + filtered_data['venue'] + ". " + filtered_data['year']
y = filtered_data.authorId
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.35, stratify = y, random_state = 8)

vect = TfidfVectorizer(min_df = 0.0002, max_df = 0.8, ngram_range = (1,2), max_features=100000)
X_train_m = vect.fit_transform(X_train)

model = MultinomialNB(alpha=0.00025)
model.fit(X_train_m, y_train)
X_validate_m = vect.transform(X_validate)

print(model.score(X_validate_m, y_validate))

####### Training model with full train.json #######

with open("train.json") as file: 
    jdata = json.load(file)

data_full = pd.DataFrame(jdata)
data_full['authorId'] = pd.to_numeric(data_full['authorId'])
data_full['year'] = data_full['year'].map(str)

X = data_full['abstract'] + ". " + data_full['title'] + ". " + data_full['venue'] + ". " + data_full['year']
y = data_full['authorId']
vect = TfidfVectorizer(min_df = 0.0002, max_df = 0.8, ngram_range = (1,2), max_features=100000)
X_train_full= vect.fit_transform(X)

model = MultinomialNB(alpha=0.00025)
model.fit(X_train_full, y)

with open("test.json") as file2: 
    test = json.load(file2)

test_data = pd.DataFrame(test)
test_data['year'] = test_data['year'].map(str)
X_test = test_data['abstract'] + ". " + test_data['title'] + ". " + test_data['venue'] + ". " + test_data['year']
X_test_m = vect.transform(X_test)
authorId = model.predict(X_test_m)
paperId = test_data.paperId

result = pd.DataFrame({'paperId':paperId, 'authorId':authorId})
result['authorId'] = result['authorId'].map(str)

nested = result.to_json(orient='records')
parsed = json.loads(nested)

with open("predicted.json", "w") as write_file:
    json.dump(parsed, write_file, indent=4)