import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='english', help="dataset language")
args = vars(parser.parse_args())

# Load from file
joblib_file = f"pretrained_model/classification_{args['language']}_model.pkl"
joblib_model = joblib.load(joblib_file)

recognitionResultsArr = []
recognitionResults = open('./Results/text.txt')

for line in recognitionResults:
    recognitionResultsArr.append(line.rstrip())

text_clf_result = joblib_model.predict(recognitionResultsArr)

resultArray = []
for index, line in enumerate(recognitionResultsArr):
    resultArray.append(f'{recognitionResultsArr[index]} - {text_clf_result[index]}')

print('Text classification result: ', resultArray)
textCategories = open('./Results/textCategories.txt', 'a')
textCategories.write(f'{resultArray}\n')