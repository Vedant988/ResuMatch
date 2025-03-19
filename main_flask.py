# app.py
from flask import Flask,render_template,request,redirect,url_for,flash,jsonify,session
import PyPDF2
import uuid
import os
import json
from pinecone import Pinecone,ServerlessSpec
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field
from sentence_transformers import SentenceTransformer

load_dotenv()
app=Flask(__name__)
app.secret_key='resume_ranking_secret_key'

UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)),'uploads')
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

class ResumeData(BaseModel):
    """Structured schema for resume extraction"""
    name: str=Field(description="Full name of the candidate")
    contact: str=Field(description="Contact information")
    skills: list=Field(description="List of professional skills")
    work_experience: list=Field(description="Detailed work experience")
    education: list=Field(description="Educational background")
    certifications: list=Field(description="Professional certifications")

class ResumeRankingSystem:
    def __init__(self):
        self.llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key= os.getenv("GOOGLE_API_KEY")
        )
        
        self.embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
        
        self.PINECONE_ENVIRONMENT="us-east-1"
        self.PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

        self.pc=Pinecone(api_key=self.PINECONE_API_KEY,environment=self.PINECONE_ENVIRONMENT)
        self.index_name="resume-matching-index"
        self.current_session_ids=[]  

        try:
            self.index_description=self.pc.describe_index(self.index_name)
            self.index=self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Index '{self.index_name}' not found: {e}")
            self.create_pinecone_index()
        
        self.resume_extraction_prompt=PromptTemplate(
    template="""
        Extract structured information from the following resume text:
        
        {resume_text}
        
        Extract the information following this JSON schema exactly:
        
        json
        {{
            "name": "Full name of candidate",
            "contact": {{
                "email": "Email address",
                "phone": "Phone number",
                "linkedin": "LinkedIn profile (if available)",
                "location": "City/State/Country"
            }},
            "skills": ["List of technical and soft skills"],
            "work_experience": [
                {{
                    "company": "Company name",
                    "role": "Job title",
                    "duration": "Employment period (e.g.,'Jan 2020 - Present')",
                    "description": "Brief description of responsibilities and achievements",
                    "technologies": ["Technologies or tools used"]
                }}
            ],
            "education": [
                {{
                    "institution": "University or school name",
                    "degree": "Degree earned (if available)",
                    "field": "Field of study/branch",
                    "graduation_date": "Graduation date or expected graduation",
                    "cgpa": "CGPA or percentage (if available)"
                }}
            ],
            "certifications": [
                {{
                    "name": "Certification name",
                    "issuer": "Issuing organization",
                    "date": "Date obtained"
                }}
            ],
            "projects": [
                {{
                    "name": "Project name",
                    "description": "Brief description",
                    "technologies": ["Technologies used"]
                }}
            ]
        }}
        
        
        Instructions:
        1. Parse the resume carefully,identifying structured sections and content.
        2. Extract only factual information that appears in the document.
        3. Leave fields empty (with empty strings or arrays) if the information is not available in the resume.
        4. Format dates consistently using month-year format (e.g.,"Jan 2020 - Mar 2022").
        5. For skills,include both technical skills (languages,frameworks,tools) and soft skills.
        6. Watch for section headings in the resume to properly categorize information.
        7. For contact information,extract all available details but only include what's actually present.
        8. If CGPA is not provided in a standard format,convert it to a numerical value on a 4.0 or 10.0 scale.
        
        Return a valid,parseable JSON object exactly matching the schema above.
        """,
        input_variables=["resume_text"]
    )
        
        self.json_parser=JsonOutputParser(pydantic_object=ResumeData)
    
    def create_pinecone_index(self):
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.PINECONE_ENVIRONMENT
                )
            )
            
            import time
            time.sleep(5)  
            
            self.index=self.pc.Index(self.index_name)
            self.index_description=self.pc.describe_index(self.index_name)
            print(f"Successfully created index: {self.index_name}")
            return True
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return False
    
    def delete_session_vectors(self):
        if not self.index:
            print("Pinecone index not initialized. Cannot delete vectors.")
            return False
            
        if not self.current_session_ids:
            print("No vectors to delete for this session.")
            return True
            
        try:
            batch_size=1000
            for i in range(0,len(self.current_session_ids),batch_size):
                batch=self.current_session_ids[i:i+batch_size]
                self.index.delete(ids=batch)
            
            print(f"Successfully deleted {len(self.current_session_ids)} vectors from index")
            self.current_session_ids=[]  
            return True
        except Exception as e:
            print(f"Error deleting vectors from Pinecone index: {e}")
            return False
    
    def descriptionEnhance(self,description):
        query_expansion_prompt=PromptTemplate(
        template="""
        You are an advanced AI specializing in semantic query expansion for vector embedding similarity search.
        
        # Task
        Transform the provided query into an expanded boolean query that will maximize recall and relevance in vector similarity search (cosine) systems like Pinecone.
        
        # Guidelines for Effective Query Expansion
        1. Break down the original query into its core concepts
        2. For each concept,generate 4-6 semantically similar terms or phrases
        3. Structure the expanded query using boolean operators:
        - Use OR between similar terms (synonyms,related concepts)
        - Use AND between different concept groups to maintain relevance
        4. Include both technical terms and common language variations
        5. Maintain the original intent while broadening the semantic reach
        
        # Bias Mitigation Guidelines
        1. Ensure expansions are balanced and do not favor particular:
        - Demographics (gender,age,ethnicity,etc.)
        - Technologies or platforms
        - Industries or sectors
        - Geographic regions
        2. Use neutral language that avoids stereotypes
        3. Include diverse terminology that represents various contexts
        4. Avoid terms with unintended negative connotations
        5. When expanding job-related queries,ensure expansions represent various levels of experience
        
        # Output Format
        Return ONLY the expanded query as a text string using boolean operators. 
        DO NOT include any explanatory text,JSON formatting,or markdown.
        
        # Example
        Original query: "Cloud infrastructure engineer with AWS experience"
        
        Expected output:
        (Cloud OR AWS OR Amazon Web Services OR infrastructure OR architecture OR platform) AND (engineer OR developer OR architect OR specialist OR administrator) AND (AWS OR Amazon Web Services OR EC2 OR S3 OR cloud provider OR cloud platform)
        
        Now expand the following query:
        ORIGINAL QUERY: {description}
        
        EXPANDED QUERY:
        """,
        input_variables=["description"]
    )

        try:
            response=self.llm.invoke(query_expansion_prompt.format(description=description))
            expanded_query=response.content.strip()
            
            if expanded_query.startswith("```") and expanded_query.endswith("```"):
                expanded_query=expanded_query.split("```")[1].strip()
            
            return expanded_query
        except Exception as e:
            print(f"Error enhancing job description: {e}")
            return description
        
    def extract_pdf_text(self,pdf_path):
        with open(pdf_path,'rb') as file:
            reader=PyPDF2.PdfReader(file)
            full_text=""
            for page in reader.pages:
                full_text += page.extract_text()
        return full_text

    def process_resume(self,resume_text):
        extraction_chain=self.resume_extraction_prompt | self.llm | self.json_parser
        
        try:
            structured_resume=extraction_chain.invoke({"resume_text": resume_text})
            return structured_resume
        except Exception as e:
            print(f"Error processing resume: {e}")
            return None

    def generate_resume_embedding(self,structured_resume):
        resume_text=" ".join([
            " ".join(structured_resume.get('skills',[])),
            " ".join(str(exp) for exp in structured_resume.get('work_experience',[])),
            " ".join(str(edu) for edu in structured_resume.get('education',[]))
        ])
        return self.embedding_model.encode(resume_text).tolist()

    def store_in_pinecone(self,resume_id,embedding,metadata):
        if not self.index:
            print("Pinecone index not initialized. Creating index...")
            success=self.create_pinecone_index()
            if not success:
                print("Failed to create Pinecone index")
                return False
        
        try:
            self.index.upsert([
                (resume_id,embedding,{
                    'original_filename': metadata['original_filename'],
                    'structured_resume': json.dumps(metadata['structured_resume']) 
                })
            ])
            self.current_session_ids.append(resume_id)  
            return True
        except Exception as e:
            print(f"Error storing in Pinecone: {e}")
            return False

    def rank_resumes(self,job_description):
        if not self.index:
            print("Pinecone index not initialized. Cannot rank resumes.")
            return []
        
        try:
            job_embedding=self.embedding_model.encode(job_description).tolist()

            results=self.index.query(
                vector=job_embedding,
                top_k=10,
                include_metadata=True
            )
            print(results)
            return results['matches']
        except Exception as e:
            print(f"Error ranking resumes: {e}")
            return []

ranking_system=ResumeRankingSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume_submission')
def resume_submission():
    ranking_system.current_session_ids=[]
    return render_template('resume_submission.html')

@app.route('/compare_resume')
def compare_resume():
    return render_template('compare.html')

@app.route('/upload_resumes',methods=['POST'])
def upload_resumes():
    if 'files[]' not in request.files:
        return jsonify({"success":False,"message":"No files part"})
    
    files=request.files.getlist('files[]')
    uploaded_files=[]
    
    for file in files:
        if file.filename=='':
            continue
            
        if file and file.filename.endswith('.pdf'):
            filename=str(uuid.uuid4())+'.pdf'
            filepath=os.path.join(UPLOAD_FOLDER,filename)
            file.save(filepath)
            
            uploaded_files.append({
                'original_name': file.filename,
                'saved_as': filename,
                'path': filepath
            })
    
    session_files=session.get('uploaded_files',[])
    session_files.extend(uploaded_files)
    session['uploaded_files']=session_files
    
    return jsonify({
        "success": True,
        "files": [f['original_name'] for f in uploaded_files]
    })

@app.route('/analyze',methods=['POST'])
def analyze_resumes():
    data=request.get_json()
    
    if not data or 'jobDescription' not in data or not data['jobDescription']:
        return jsonify({"success": False,"message": "Missing job description"})
    
    if 'files' not in data or not data['files']:
        return jsonify({"success": False,"message": "No resume files provided"})
    
    # Delete previous vectors for this session before adding new ones
    # delete_success=ranking_system.delete_session_vectors()
    # if not delete_success:
    #     print("Warning: Failed to delete previous session vectors")
    
    job_description=data['jobDescription']
    resume_files=data['files']
    
    processed_files=[]
    
    for file_info in resume_files:
        try:
            filename=file_info['filename']
            file_path=os.path.join(UPLOAD_FOLDER,filename)            
            raw_text=ranking_system.extract_pdf_text(file_path)            
            structured_resume=ranking_system.process_resume(raw_text)
            
            if structured_resume:
                resume_id=str(uuid.uuid4())
                embedding=ranking_system.generate_resume_embedding(structured_resume)
                
                success=ranking_system.store_in_pinecone(
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
    
    ranked_resumes=ranking_system.rank_resumes(job_description)
    
    result=[]
    unique_resumes={}
    
    for match in ranked_resumes:
        filename=match['metadata']['original_filename']
        if filename not in unique_resumes:
            unique_resumes[filename]={
                'filename': filename,
                'score': match['score'],
                'resume_data': json.loads(match['metadata']['structured_resume'])
            }
    
    result=list(unique_resumes.values())
    result.sort(key=lambda x: x['score'],reverse=True)
    
    return jsonify({
        "success": True,
        "ranked_resumes": result
    })

@app.route('/process_uploads',methods=['POST'])
def process_uploads():
    print("=== PROCESS_UPLOADS FUNCTION CALLED ===")
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"Form data keys: {list(request.form.keys())}")
    print(f"Files keys: {list(request.files.keys())}")
    
    if 'files' not in request.files:
        print("ERROR: 'files' not in request.files")
        print(f"Available keys in request.files: {list(request.files.keys())}")
        flash('No file part','error')
        return redirect(request.url)
        
    files=request.files.getlist('files')
    print(f"Number of files received: {len(files)}")
    for i,file in enumerate(files):
        print(f"File {i+1}: {file.filename}")
    
    job_description=request.form.get('jobDescription','')
    print(f"Job Description length: {len(job_description)}")
    print(f"Job Description preview: {job_description[:100]}...")

    
    if not job_description:
        print("ERROR: Job description is empty")
        flash('Job description is required','error')
        return redirect(request.url)
    

    job_description=ranking_system.descriptionEnhance(job_description)
    print(f"Enhanced Job Description: {job_description}")


    print("Deleting previous session vectors")
    delete_success=ranking_system.delete_session_vectors()
    if not delete_success:
        print("WARNING: Failed to delete previous session vectors")
    print("Starting fresh session")
    ranking_system.current_session_ids=[]  
        
    processed_resumes=[]
    
    for file in files:
        if file.filename == '':
            print(f"Skipping empty filename")
            continue
            
        if file and file.filename.endswith('.pdf'):
            print(f"Processing file: {file.filename}")
            with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp:
                file.save(temp.name)
                temp_path=temp.name
                print(f"Saved to temp path: {temp_path}")
            
            try:
                print(f"Extracting text from {file.filename}")
                raw_text=ranking_system.extract_pdf_text(temp_path)
                print(f"Extracted text length: {len(raw_text)}")
                
                print(f"Processing resume with LLM")
                structured_resume=ranking_system.process_resume(raw_text)
                
                if structured_resume:
                    print(f"Resume processed successfully")
                    resume_id=str(uuid.uuid4())
                    print(f"Generated resume ID: {resume_id}")
                    embedding=ranking_system.generate_resume_embedding(structured_resume)
                    print(f"Generated embedding of length: {len(embedding)}")
                    
                    print(f"Storing in Pinecone")
                    success=ranking_system.store_in_pinecone(
                        resume_id,
                        embedding,
                        metadata={
                            'original_filename': file.filename,
                            'structured_resume': structured_resume
                        }
                    )
                    if success:
                        processed_resumes.append({
                            'id': resume_id,
                            'filename': file.filename,
                            'resume_data': structured_resume
                        })
                        print(f"Added resume to processed list")
                    else:
                        print(f"ERROR: Failed to store resume in Pinecone")
                else:
                    print(f"ERROR: Failed to process resume structure")
            except Exception as e:
                print(f"ERROR processing {file.filename}: {str(e)}")
                flash(f'Error processing {file.filename}: {str(e)}','error')
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Removed temp file: {temp_path}")
    print(f"Total processed resumes: {len(processed_resumes)}")

    print(f"Ranking resumes against job description")
    ranked_resumes=ranking_system.rank_resumes(job_description)
    print(f"Got {len(ranked_resumes)} ranked results")
    
    results=[]
    for match in ranked_resumes:
        try:
            metadata=match.get('metadata',{})
            filename=metadata.get('original_filename','Unknown file')
            print(f"Processing match for file: {filename}")
            
            try:
                if 'structured_resume' in metadata:
                    if isinstance(metadata['structured_resume'],str):
                        resume_data=json.loads(metadata['structured_resume'])
                    else:
                        resume_data=metadata['structured_resume']
                else:
                    resume_data={'name': 'Unknown','contact': 'No contact info',
                                   'skills': [],'work_experience': [],
                                   'education': [],'certifications': []}
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse JSON for {filename}")
                resume_data={'name': 'Error parsing resume','contact': '',
                              'skills': [],'work_experience': [],
                              'education': [],'certifications': []}
            
            results.append({
                'filename': filename,
                'score': match.get('score',0),
                'resume_data': resume_data
            })
            print(f"Added {filename} to results with score {match.get('score',0)}")
        except Exception as e:
            print(f"ERROR processing match: {e}")
            continue
    
    print(f"Final results count: {len(results)}")
    print("=== PROCESS_UPLOADS FUNCTION COMPLETED ===")


    for i,result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Filename: {result['filename']}")
        print(f"  Score: {result['score']}")
        print(f"  Name: {result['resume_data'].get('name','No name found')}")
        if isinstance(result['resume_data'],dict):
            print(f"  Resume data keys: {list(result['resume_data'].keys())}")
        else:
            print(f"  Resume data is not a dictionary,it's a {type(result['resume_data'])}")

    return render_template('results.html',results=results)


@app.route('/compare_candidates',methods=['POST'])
def compare_candidates():
    if len(request.files) < 2:
        return jsonify({"success": False,"message": "Two resumes are required for comparison"})
    
    resume1=request.files.get('resume1')
    resume2=request.files.get('resume2')
    
    if not resume1 or not resume2:
        return jsonify({"success": False,"message": "Both resumes are required"})
    
    if not resume1.filename.endswith('.pdf') or not resume2.filename.endswith('.pdf'):
        return jsonify({"success": False,"message": "Only PDF files are supported"})
    
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp1:
        resume1.save(temp1.name)
        temp_path1=temp1.name
    
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp2:
        resume2.save(temp2.name)
        temp_path2=temp2.name
    
    try:
        text1=ranking_system.extract_pdf_text(temp_path1)
        text2=ranking_system.extract_pdf_text(temp_path2)
        
        resume_data1=ranking_system.process_resume(text1)
        resume_data2=ranking_system.process_resume(text2)
        
        if not resume_data1 or not resume_data2:
            return jsonify({"success": False,"message": "Failed to process one or both resumes"})
        
        comparison_result=generate_comparison(resume_data1,resume_data2)
        
        return render_template('comparison_result.html',
                              candidate1=resume_data1,
                              candidate2=resume_data2,
                              result=comparison_result)
    
    except Exception as e:
        return jsonify({"success": False,"message": f"Error comparing resumes: {str(e)}"})
    finally:
        if os.path.exists(temp_path1):
            os.unlink(temp_path1)
        if os.path.exists(temp_path2):
            os.unlink(temp_path2)

def generate_comparison(resume1,resume2):
    """Generate comparison between two resumes using LLM"""
    compare_prompt=PromptTemplate(
        template="""
        You are a professional HR evaluator comparing two candidate resumes.
        
        First Resume: {resume1}
        Second Resume: {resume2}
        
        Compare these two resumes comprehensively and determine which candidate is stronger. Analyze their projects,work experience,academic scores,and identify the ‘Strong Zone Area’ based on their project work,skills,and expertise. Assess how their experiences and skills align with their personal strengths,focusing on project involvement and the impact of their work in key areas
        
        Provide your analysis in the following JSON format:
        {{
            "winner": "Candidate 1" or "Candidate 2",
            "overall_score": {{"candidate1": score_0_100,"candidate2": score_0_100}},
            "technical_skills_comparison": "detailed comparison of skills...",
            "experience_comparison": "detailed comparison of work experience...",
            "education_comparison": "detailed comparison of education...",
            "candidate1_strengths": ["strength1","strength2","strength3"],
            "candidate1_weaknesses": ["weakness1","weakness2"],
            "candidate2_strengths": ["strength1","strength2","strength3"],
            "candidate2_weaknesses": ["weakness1","weakness2"],
            "verdict": "detailed final verdict explaining your choice..."
        }}
        """,
        input_variables=["resume1","resume2"]
    )

    json_parser=JsonOutputParser()
    
    comparison_chain=compare_prompt | ranking_system.llm | json_parser
    
    try:
        result=comparison_chain.invoke({
            "resume1": json.dumps(resume1),
            "resume2": json.dumps(resume2)
        })
        
        required_fields=[
            "winner","overall_score","technical_skills_comparison",
            "experience_comparison","education_comparison",
            "candidate1_strengths","candidate1_weaknesses",
            "candidate2_strengths","candidate2_weaknesses","verdict"
        ]
        
        for field in required_fields:
            if field not in result:
                if field == "overall_score":
                    result[field]={"candidate1": 50,"candidate2": 50}
                elif field.endswith("_strengths") or field.endswith("_weaknesses"):
                    result[field]=["Not specified"]
                else:
                    result[field]="Not analyzed"
        
        return result
    except Exception as e:
        print(f"Error in comparison: {e}")
        return {
            "winner": "Unable to determine",
            "overall_score": {"candidate1": 50,"candidate2": 50},
            "technical_skills_comparison": "Error processing comparison",
            "experience_comparison": "Error processing comparison",
            "education_comparison": "Error processing comparison",
            "candidate1_strengths": ["Not analyzed"],
            "candidate1_weaknesses": ["Not analyzed"],
            "candidate2_strengths": ["Not analyzed"],
            "candidate2_weaknesses": ["Not analyzed"],
            "verdict": "Error processing comparison"
        }
    
if __name__ == '__main__':
    app.run(debug=True)