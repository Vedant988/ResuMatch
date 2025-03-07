import streamlit as st
import PyPDF2
import uuid
import os
from pinecone_reset import Pinecone
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from sentence_transformers import SentenceTransformer

class ResumeData(BaseModel):
    """Structured schema for resume extraction"""
    name: str = Field(description="Full name of the candidate")
    contact: str = Field(description="Contact information")
    skills: list = Field(description="List of professional skills")
    work_experience: list = Field(description="Detailed work experience")
    education: list = Field(description="Educational background")
    certifications: list = Field(description="Professional certifications")

class ResumeRankingSystem:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key= "AIzaSyAPNdgdFGnD7TrhwyGyZTEDD4dhfl9Epys"
        )
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.PINECONE_ENVIRONMENT = "us-east-1"
        self.PINECONE_API_KEY = "pcsk_4D2ykq_A7TRYdCkTgkamSc9JtuVFhM3NWERngzs6KkUq1aGwEKyVEbsxm8V91f99xeP6yr"

        self.pc = Pinecone(api_key=self.PINECONE_API_KEY, environment=self.PINECONE_ENVIRONMENT)
        self.index_name = "resume-matching-index"

        try:
            self.index_description = self.pc.describe_index(self.index_name)
            self.index = self.pc.Index(self.index_name)
        except pinecone.exceptions.NotFoundException:
            st.error(f"Index '{self.index_name}' not found. Please create it in Pinecone first.")
            self.index = None
            self.index_description = None
        
        self.resume_extraction_prompt = PromptTemplate(
            template="""
            Extract structured information from the following resume text:
            
            {resume_text}
            
            Extract the information following this JSON schema:
            - Provide full name
            - Include contact details
            - List all professional skills
            - Detail work experience with companies and roles
            - List educational background including CGPA or CG
            - Note any professional certifications
            
            Ensure the response is a valid JSON that can be parsed directly.
            """,
            input_variables=["resume_text"]
        )
        
        self.json_parser = JsonOutputParser(pydantic_object=ResumeData)

    def extract_pdf_text(self, pdf_file):
        """Extract text from PDF"""
        reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
        return full_text

    def process_resume(self, resume_text):
        extraction_chain = self.resume_extraction_prompt | self.llm | self.json_parser
        
        try:
            structured_resume = extraction_chain.invoke({"resume_text": resume_text})
            return structured_resume
        except Exception as e:
            st.error(f"Error processing resume: {e}")
            return None

    def generate_resume_embedding(self, structured_resume):
        """Generate embedding for resume"""
        resume_text = " ".join([
            " ".join(structured_resume.get('skills', [])),
            " ".join(str(exp) for exp in structured_resume.get('work_experience', [])),
            " ".join(str(edu) for edu in structured_resume.get('education', []))
        ])
        return self.embedding_model.encode(resume_text).tolist()

    def store_in_pinecone(self, resume_id, embedding, metadata):
        """Store resume in Pinecone vector database"""
        if self.index:
            self.index.upsert([
                (resume_id, embedding, {
                    'original_filename': metadata['original_filename'],
                    'structured_resume': json.dumps(metadata['structured_resume']) 
                })
            ])
        else:
            st.error("Pinecone index not initialized. Cannot store resume.")

    def rank_resumes(self, job_description):
        """Rank resumes based on job description"""
        if self.index:
            job_embedding = self.embedding_model.encode(job_description).tolist()

            results = self.index.query(
                vector=job_embedding, 
                top_k=10,
                include_metadata=True
            )

            return results['matches']
        else:
            st.error("Pinecone index not initialized. Cannot rank resumes.")
            return []

def main():
    st.title("🚀 AI-Powered Resume Ranking System")
    ranking_system = ResumeRankingSystem()
    st.sidebar.header("📤 Upload Resumes")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF resumes",
        type=['pdf'],
        accept_multiple_files=True
    )

    if st.sidebar.button("Process Resumes"):  
        if uploaded_files:
            with st.spinner('Processing Resumes...'):
                processed_resumes = []
                for uploaded_file in uploaded_files:
                    resume_id = str(uuid.uuid4())
                    raw_text = ranking_system.extract_pdf_text(uploaded_file)

                    structured_resume = ranking_system.process_resume(raw_text)

                    if structured_resume:
                        embedding = ranking_system.generate_resume_embedding(structured_resume)

                        ranking_system.store_in_pinecone(
                            resume_id,
                            embedding,
                            metadata={
                                'original_filename': uploaded_file.name,
                                'structured_resume': structured_resume
                            }
                        )

                        processed_resumes.append({
                            'id': resume_id,
                            'filename': uploaded_file.name,
                            'resume_data': structured_resume
                        })

                st.sidebar.success(f"Processed {len(processed_resumes)} resumes")
        else:
            st.sidebar.warning("Please upload resumes before processing.")

    st.header("📋 Job Description")
    job_description = st.text_area(
        "Enter Detailed Job Description",
        height=200
    )

    if st.button("🔍 Rank Resumes"):
        if not job_description:
            st.error("Please enter a job description")
        else:
            with st.spinner('Ranking Resumes...'):
                ranked_resumes = ranking_system.rank_resumes(job_description)

                unique_resumes = {}
                for match in ranked_resumes:
                    filename = match['metadata']['original_filename']
                    if filename not in unique_resumes:
                        unique_resumes[filename] = match

                ranked_resumes = list(unique_resumes.values())

                st.header("🏆 Ranked Resumes")
                for i, match in enumerate(ranked_resumes, 1):
                    st.subheader(f"Rank {i}")
                    st.write(f"Match Score: {match['score']:.2f}")

                    metadata = match['metadata']
                    st.write(f"Filename: {metadata['original_filename']}")

                    structured_resume_json = metadata['structured_resume']
                    structured_resume = json.loads(structured_resume_json)
                    with st.expander("Resume Details"):
                        st.json(structured_resume)

if __name__ == "__main__":
    main()