import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

#Constants
GREETING_INPUTS = ["Hello", "Hi", "Greetings", "What's up", "Hey", "Hiii", "Hii"]
CONCLUDE_INPUTS = ["Thanks", "Welcome"]
ACKNOWLEDGEMENT = ["OK", "Okay"]


def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    cleaned_words = [w for w in word_tok if not w in stop_words]
    return ' '.join(cleaned_words)

def get_response(usrText):
    if usrText.lower() == "bye":
        return "Bye"

    a = [x.lower() for x in GREETING_INPUTS]
    c = [x.lower() for x in ACKNOWLEDGEMENT]
    d = [x.lower() for x in CONCLUDE_INPUTS]
    
    if usrText.lower() in a:
        return ("Hi, I'm JARVIS!")

    if usrText.lower() in c:
        return ("Alright!")

    if usrText.lower() in d:
        return ("My pleasure!")


    t_usr = tfv.transform([cleanup(usrText.lower())])
    category = le.inverse_transform(model.predict(t_usr))

    category_dataset = data[data['Class'].values == category]

    cos_sims = []
    for question in category_dataset['Question']:
        sims = cosine_similarity(tfv.transform([question]), t_usr)

        cos_sims.append(sims)

    ind = cos_sims.index(max(cos_sims))

    b = [category_dataset.index[ind]]

    if max(cos_sims) > [[0.]]:
        a = data['Answer'][category_dataset.index[ind]]+"   "
        return a


    elif max(cos_sims)==[[0.]]:
       return "Sorry! I couldn't understand the query!"


if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    
    le = LabelEncoder()
    tfv = TfidfVectorizer(min_df=1, stop_words='english')
    data = pd.read_csv("C:\IBM\Banking-Chatbot-master\BankFAQs.csv")
    questions = data['Question'].values
    X = []

    for question in questions:
        X.append(cleanup(question))

    X = tfv.fit_transform(X)

    y = le.fit_transform(data['Class'])

    trainx, testx, trainy, testy = train_test_split(X, y, test_size=.3, random_state=38)

    model = SVC(kernel='linear')
    model.fit(trainx, trainy)
    y_pred = model.predict(testx)
    
    test_accuracy = accuracy_score(testy, y_pred)
    print(f'Accuracy Score is {test_accuracy}')
  
    userInput = ""
    
   
    print("Type 'exit' or 'quit' to exit the chat:\n")
    
    while True:
        if userInput == "":
            print("JARVIS: Hi, Im JARVIS. Please enter your query related to bank.\n\n")
        
        userInput = input("User: ")
        
        if userInput.lower() == "exit" or userInput.lower() == "quit":
            break
        else:
            response = get_response(userInput)
            print("JARVIS: "+response+"\n\n")
    
    print("Thanks, you can check any queries related to bank anytime!")