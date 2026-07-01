import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
df = pd.read_csv('UpdatedResumeDataSet.csv')
df.head()
df.shape
plt.figure(figsize=(15,5))
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()
df['Category'].unique()
counts=df['Category'].value_counts()
labels=df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()
df['Category'][0]
df['Resume'][0]
#checking the original category distribution
print("Original Category Distribution")
print(df['Category'].value_counts())
#get largest category size(i.e. category size with hihest number of entries)
max_size = df['Category'].value_counts().max()
#perform oversampling
balanced_df =df.groupby('Category').apply(lambda x: x.sample(max_size ,replace=True)).reset_index(drop=True)
# shuffle the dataset to otrder any bias
df = balanced_df.sample(frac=1).reset_index(drop=True)
#check the balanced categoryt distributiuon
print("\nbalanced Category Distribution(after Oversampling):")
print(df['Category'].value_counts())
import re
def cleanResume(txt):
    cleanText=re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText
cleanResume("my #### $ #  #riteshpujari webiste like is this http://heloword and access it @gmain.com")
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
df['Resume'][0]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
df.Category.unique()
# ['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
#        'Mechanical Engineer', 'Sales', 'Health and fitness',
#        'Civil Engineer', 'Java Developer', 'Business Analyst',
#        'SAP Developer', 'Automation Testing', 'Electrical Engineering',
#        'Operations Manager', 'Python Developer', 'DevOps Engineer',
#        'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
#        'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing'],
#       dtype=object)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(df['Resume'])
requredTaxt  = tfidf.transform(df['Resume'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)
df['Category'].value_counts()
X_train.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure that X_train and X_test are dense if they are sparse
X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# 1. Train KNeighborsClassifier
knn_model = OneVsRestClassifier(KNeighborsClassifier())
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("\nKNeighborsClassifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_knn)}")
#2 Train SVc model]
svc_model = OneVsRestClassifier(SVC())
svc_model.fit(X_train,y_train)
y_pred_svc = svc_model.predict(X_test)
print("\nSVC Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svc)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_svc)}")
# 3. Train RandomForestClassifier
rf_model = OneVsRestClassifier(RandomForestClassifier())
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandomForestClassifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")
import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(svc_model, open('clf.pkl', 'wb'))
pickle.dump(le, open("encoder.pkl",'wb'))
# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume) 

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])
    
    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name

# pred(myresume)  = """
# Jane Smith is a certified personal trainer with over 5 years of experience in helping individuals achieve their fitness goals. Specializing in weight loss, strength training, and sports conditioning, Jane has developed personalized workout routines for clients of all ages and fitness levels. She has extensive knowledge in nutrition and exercise science, and uses this to create holistic health and fitness programs that are tailored to individual needs.

# Jane holds a degree in Exercise Science and is a certified trainer through the National Academy of Sports Medicine (NASM). She has worked with athletes, seniors, and individuals with chronic health conditions, helping them improve their physical well-being and overall quality of life.

# Her expertise includes:
# - Weight Loss and Body Composition
# - Strength Training and Resistance Exercises
# - Cardio Conditioning
# - Nutrition Coaching and Meal Planning
# - Injury Prevention and Rehabilitation
# - Functional Movement and Flexibility Training
# - Group Fitness Classes

# Certifications:
# - Certified Personal Trainer, NASM
# - CPR and First Aid Certified
# - Yoga Instructor (200-Hour Certification)

# Education:
# BSc in Exercise Science, ABC University, 2014-2018

# Work Experience:
# - Personal Trainer at XYZ Fitness Gym (2018-Present)
# - Fitness Coach at Wellness Center (2016-2018)

# Languages:
# - English (Fluent)
# - Spanish (Conversational)
# """

# # Now, test the model with the Health and Fitness-focused resume
# pred(myresume)


# myresume = """
# John Doe is an experienced Network Security Engineer with over 7 years of expertise in designing, implementing, and managing network security infrastructures. Specializing in safeguarding critical network systems, John has worked with various organizations to protect against cyber threats, data breaches, and unauthorized access. He is proficient in deploying firewalls, intrusion detection systems (IDS), VPNs, and network monitoring tools to ensure the integrity and security of networks.

# John holds a degree in Computer Science and certifications in several cybersecurity domains, including Certified Information Systems Security Professional (CISSP), Certified Ethical Hacker (CEH), and Cisco Certified Network Associate (CCNA). He has extensive experience in troubleshooting and resolving network vulnerabilities, and has played a key role in conducting security audits and risk assessments.

# Key Skills:
# - Network Security Architecture
# - Firewall Management and Configuration
# - Intrusion Detection and Prevention Systems (IDS/IPS)
# - Virtual Private Networks (VPNs)
# - Security Audits and Risk Assessments
# - Cybersecurity Incident Response
# - Network Monitoring and Traffic Analysis
# - Vulnerability Assessment and Penetration Testing
# - Data Encryption and Secure Communications

# Certifications:
# - CISSP (Certified Information Systems Security Professional)
# - CEH (Certified Ethical Hacker)
# - CCNA (Cisco Certified Network Associate)
# - CompTIA Security+

# Education:
# BSc in Computer Science, XYZ University, 2012-2016

# Professional Experience:
# - Network Security Engineer at ABC Corp (2016-Present)
# - IT Security Specialist at DEF Solutions (2014-2016)

# Languages:
# - English (Fluent)
# - French (Intermediate)
# """

# # Now, test the model with the Network Security Engineer-focused resume
# pred(myresume)

# myresume = """
# Sarah Williams is a dedicated and skilled advocate with over 10 years of experience in providing legal representation to clients across various sectors, including criminal law, civil litigation, and family law. With a deep understanding of legal procedures and case law, Sarah has successfully handled numerous cases in the courtroom, negotiating favorable settlements and providing expert legal advice to individuals and businesses.

# She holds a law degree from XYZ University and is a licensed attorney, practicing law in multiple jurisdictions. Sarah is passionate about ensuring justice is served and strives to make legal processes more accessible to her clients. She is known for her excellent research and analytical skills, attention to detail, and commitment to upholding the law with integrity.

# Key Skills:
# - Criminal Law
# - Civil Litigation
# - Family Law
# - Contract Law
# - Legal Research and Writing
# - Courtroom Advocacy
# - Legal Counseling and Advice
# - Client Relationship Management
# - Legal Compliance and Regulations
# - Negotiation and Settlement

# Certifications and Licenses:
# - Licensed Attorney at Law, XYZ State Bar
# - Certification in Criminal Law, XYZ University

# Education:
# JD in Law, XYZ University, 2010-2013

# Professional Experience:
# - Senior Advocate at ABC Law Firm (2016-Present)
# - Associate Advocate at DEF Legal Group (2013-2016)

# Languages:
# - English (Fluent)
# - Spanish (Conversational)
# """

# # Now, test the model with the Advocate-focused resume
# pred(myresume)

