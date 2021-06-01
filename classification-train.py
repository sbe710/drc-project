from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='english', help="dataset language")
args = vars(parser.parse_args())

textData = []
texts_labels = []

f = open(f"classification-datasets/{args['language']}/merged.txt")

for line in f:
    textData.append(line.split(',')[0])
    texts_labels.append(line.split(',')[1])

texts_labels = [line.rstrip() for line in texts_labels]

print('Classification train process started')

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])
text_clf.fit(textData, texts_labels)

print('Classification train process ended')

# Save to file in the current working directory
joblib_file = f"pretrained_model/classification_{args['language']}_model.pkl"
joblib.dump(text_clf, joblib_file)

print('Classification trained model successfully saved')
