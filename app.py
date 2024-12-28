import nltk
import pickle
import re
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7F]', ' ', clean_text)    
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

def main():
    st.title("Resume Screening App")
    input_file = st.file_uploader('Upload Resume', type=['pdf', 'txt'])

    if input_file is not None:
        try:
            resume_byte = input_file.read()
            resume_text = resume_byte.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_byte.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_num = clf.predict(input_features)[0]
        st.write(prediction_num)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }

        category_name = category_mapping.get(prediction_num, "Unknown")
        st.write(category_name)

if __name__ == "__main__":
    main()

