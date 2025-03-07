# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import PyPDF2
import uuid
import os
import json
from pinecone import Pinecone
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = 'resume_ranking_secret_key'

# Create the uploads folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        except Exception as e:
            print(f"Index '{self.index_name}' not found: {e}")
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

    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
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
            print(f"Error processing resume: {e}")
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
            return True
        else:
            print("Pinecone index not initialized. Cannot store resume.")
            return False

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
            print("Pinecone index not initialized. Cannot rank resumes.")
            return []

# Initialize the ranking system
ranking_system = ResumeRankingSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume_submission')
def resume_submission():
    return render_template('resume_submission.html')

@app.route('/upload_resumes', methods=['POST'])
def upload_resumes():
    if 'files[]' not in request.files:
        return jsonify({"success": False, "message": "No files part"})
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and file.filename.endswith('.pdf'):
            filename = str(uuid.uuid4()) + '.pdf'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            uploaded_files.append({
                'original_name': file.filename,
                'saved_as': filename,
                'path': filepath
            })
    
    # Store uploaded files in session
    session_files = session.get('uploaded_files', [])
    session_files.extend(uploaded_files)
    session['uploaded_files'] = session_files
    
    return jsonify({
        "success": True, 
        "files": [f['original_name'] for f in uploaded_files]
    })

@app.route('/analyze', methods=['POST'])
def analyze_resumes():
    data = request.get_json()
    
    if not data or 'jobDescription' not in data or not data['jobDescription']:
        return jsonify({"success": False, "message": "Missing job description"})
    
    if 'files' not in data or not data['files']:
        return jsonify({"success": False, "message": "No resume files provided"})
    
    job_description = data['jobDescription']
    resume_files = data['files']
    
    processed_files = []
    
    # Process each resume
    for file_info in resume_files:
        try:
            # Get the file path
            filename = file_info['filename']
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Extract text from PDF
            raw_text = ranking_system.extract_pdf_text(file_path)
            
            # Process resume to extract structured data
            structured_resume = ranking_system.process_resume(raw_text)
            
            if structured_resume:
                # Generate embedding
                resume_id = str(uuid.uuid4())
                embedding = ranking_system.generate_resume_embedding(structured_resume)
                
                # Store in Pinecone
                success = ranking_system.store_in_pinecone(
                    resume_id,
                    embedding,
                    metadata={
                        'original_filename': file_info['original_name'],
                        'structured_resume': structured_resume
                    }
                )
                
                if success:
                    processed_files.append({
                        'id': resume_id,
                        'filename': file_info['original_name'],
                        'resume_data': structured_resume
                    })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    # Rank resumes
    ranked_resumes = ranking_system.rank_resumes(job_description)
    
    # Format the response
    result = []
    unique_resumes = {}
    
    for match in ranked_resumes:
        filename = match['metadata']['original_filename']
        if filename not in unique_resumes:
            unique_resumes[filename] = {
                'filename': filename,
                'score': match['score'],
                'resume_data': json.loads(match['metadata']['structured_resume'])
            }
    
    result = list(unique_resumes.values())
    result.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({
        "success": True,
        "ranked_resumes": result
    })
@app.route('/process_uploads', methods=['POST'])
def process_uploads():
    print("=== PROCESS_UPLOADS FUNCTION CALLED ===")
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"Form data keys: {list(request.form.keys())}")
    print(f"Files keys: {list(request.files.keys())}")
    
    if 'files' not in request.files:
        print("ERROR: 'files' not in request.files")
        print(f"Available keys in request.files: {list(request.files.keys())}")
        flash('No file part', 'error')
        return redirect(request.url)
        
    files = request.files.getlist('files')
    print(f"Number of files received: {len(files)}")
    for i, file in enumerate(files):
        print(f"File {i+1}: {file.filename}")
    
    job_description = request.form.get('jobDescription', '')
    print(f"Job Description length: {len(job_description)}")
    print(f"Job Description preview: {job_description[:100]}...")
    
    if not job_description:
        print("ERROR: Job description is empty")
        flash('Job description is required', 'error')
        return redirect(request.url)
        
    processed_resumes = []
    
    # Process each resume
    for file in files:
        if file.filename == '':
            print(f"Skipping empty filename")
            continue
            
        if file and file.filename.endswith('.pdf'):
            print(f"Processing file: {file.filename}")
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
                file.save(temp.name)
                temp_path = temp.name
                print(f"Saved to temp path: {temp_path}")
            
            try:
                # Extract text from PDF
                print(f"Extracting text from {file.filename}")
                raw_text = ranking_system.extract_pdf_text(temp_path)
                print(f"Extracted text length: {len(raw_text)}")
                
                # Process resume
                print(f"Processing resume with LLM")
                structured_resume = ranking_system.process_resume(raw_text)
                
                if structured_resume:
                    print(f"Resume processed successfully")
                    # Generate embedding
                    resume_id = str(uuid.uuid4())
                    print(f"Generated resume ID: {resume_id}")
                    embedding = ranking_system.generate_resume_embedding(structured_resume)
                    print(f"Generated embedding of length: {len(embedding)}")
                    
                    # Store in Pinecone
                    print(f"Storing in Pinecone")
                    ranking_system.store_in_pinecone(
                        resume_id,
                        embedding,
                        metadata={
                            'original_filename': file.filename,
                            'structured_resume': structured_resume
                        }
                    )
                    
                    processed_resumes.append({
                        'id': resume_id,
                        'filename': file.filename,
                        'resume_data': structured_resume
                    })
                    print(f"Added resume to processed list")
                else:
                    print(f"ERROR: Failed to process resume structure")
            except Exception as e:
                print(f"ERROR processing {file.filename}: {str(e)}")
                flash(f'Error processing {file.filename}: {str(e)}', 'error')
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Removed temp file: {temp_path}")
    
    print(f"Total processed resumes: {len(processed_resumes)}")
    
    # Rank resumes
    print(f"Ranking resumes against job description")
    ranked_resumes = ranking_system.rank_resumes(job_description)
    print(f"Got {len(ranked_resumes)} ranked results")
    
    # Format for template - FIX THE ERROR HERE
    results = []
    for match in ranked_resumes:
        try:
            # Use get() method with default values to avoid KeyError
            metadata = match.get('metadata', {})
            filename = metadata.get('original_filename', 'Unknown file')
            print(f"Processing match for file: {filename}")
            
            # Parse the structured resume data, with error handling
            try:
                if 'structured_resume' in metadata:
                    if isinstance(metadata['structured_resume'], str):
                        resume_data = json.loads(metadata['structured_resume'])
                    else:
                        resume_data = metadata['structured_resume']
                else:
                    resume_data = {'name': 'Unknown', 'contact': 'No contact info', 
                                   'skills': [], 'work_experience': [], 
                                   'education': [], 'certifications': []}
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse JSON for {filename}")
                resume_data = {'name': 'Error parsing resume', 'contact': '', 
                              'skills': [], 'work_experience': [], 
                              'education': [], 'certifications': []}
            
            results.append({
                'filename': filename,
                'score': match.get('score', 0),
                'resume_data': resume_data
            })
            print(f"Added {filename} to results with score {match.get('score', 0)}")
        except Exception as e:
            print(f"ERROR processing match: {e}")
            # Skip this match if there was an error
            continue
    
    print(f"Final results count: {len(results)}")
    print("=== PROCESS_UPLOADS FUNCTION COMPLETED ===")
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

