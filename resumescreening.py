import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
model=pickle.load(open('model.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
def clean(text):
    # Remove URLs
    clean_text = re.sub(r'http\S+', ' ', text)
    
    # Remove mentions (e.g., @username)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    
    # Remove hashtags (e.g., #hashtag or standalone #)
    clean_text = re.sub(r'#\S*', ' ', clean_text)
    
    # Remove extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Remove special characters
    clean_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', clean_text)
    
    # Remove non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    
    # Final removal of extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text
category_mapping={
15:'Java Developer',
23:'Testing',
8:'DevOps Engineer',
20:'Python Developer',
24:'Web Designing',
12:'HR',
13:'Hadoop',
3:'Blockchain',
10:'ETL Developer',
18:'Operations Manager',
6:'Data Science',
22:'Sales',
16:'Mechanical Engineer',
1:'Arts',
7:'Database',
11:'Electrical Engineering',
14:'Health and fitness',
19:'PMO',
4:'Business Analyst',
9:'DotNet Developer',
2:'Automation Testing',
17:'Network Security Engineer',
21:'SAP Developer',
5:'Civil Engineer',
0:'Advocate'      





}
def main():
    st.title('Resume Screening App')
    upload_file=st.file_uploader('Upload file',type=['pdf','text','docx','csv'])
# this is check for the checking of the text format 
    if upload_file is not None:
        try:
            resume_bite=upload_file.read()
            resume_text=resume_bite.decode('utf-8')
        except UnicodeDecodeError:
            # if unicode decoder utf-8 is failed to decode try to convert it into latin-1
            resume_text=resume_bite.decode('latin-1')
        #now we give the resume_text variable to our clean function that is made above 
        clean_resume = clean(resume_text)  # Passing a list instead of a string

        vecotried=tfidf.transform([clean_resume])
        prdiction_id=model.predict(vecotried)[0]
        #st.write(prdiction_id)
        #category mapping
        #category mapping
        category_name = category_mapping.get(prdiction_id, 'Unknown')
        st.write('The Prediction ID is:',prdiction_id)
        st.write('Predicted Category:', category_name)
        

if __name__ == '__main__':
    main()
