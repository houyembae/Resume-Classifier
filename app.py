import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os

model_path = "models"
clf = pickle.load(open(os.path.join(model_path, 'clf.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(model_path, 'tfidf.pkl'), 'rb'))
le = pickle.load(open(os.path.join(model_path, 'encoder.pkl'), 'rb'))

# Resume cleaning
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# File parsers
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

# Predection
def predict_category(resume_text):
    cleaned = clean_resume(resume_text)
    vectorized = tfidf.transform([cleaned])
    prediction = clf.predict(vectorized)
    return le.inverse_transform(prediction)[0]


def main():
    st.set_page_config(page_title="Resume Category Predictor", page_icon="📄", layout="wide")
    st.title("📄 Resume Category Prediction")
    st.markdown("Upload a **PDF**, **DOCX**, or **TXT** resume to get its predicted job category.")

    uploaded_file = st.file_uploader("Upload your resume file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif ext == "docx":
                resume_text = extract_text_from_docx(uploaded_file)
            elif ext == "txt":
                resume_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return

            st.success("✅ Resume text extracted successfully!")

            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Text", resume_text, height=300)

            category = predict_category(resume_text)
            st.subheader("🎯 Predicted Job Category:")
            st.success(f"**{category}**")

        except Exception as e:
            st.error(f" Error: {str(e)}")

if __name__ == "__main__":
    main()
