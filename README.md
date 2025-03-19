# ResuMatch - Resume Comparison and Ranking System

## Overview
ResuMatch is an AI-powered resume analysis and comparison tool that helps recruiters and hiring managers efficiently evaluate candidate resumes against job descriptions and compare multiple candidates. The system uses natural language processing and vector similarity to rank resumes based on how well they match specific job requirements.

## Features
- **Resume Analysis**: Extract structured information from resume PDFs
- **Resume Ranking**: Rank multiple resumes against a job description
- **Candidate Comparison**: Compare two candidates side-by-side with detailed analysis
- **Visualization**: View comparison results with interactive charts and visual elements
- **Detailed Reports**: Get comprehensive analysis of candidate strengths and weaknesses

## Technology Stack
- **Backend**: Python, Flask
- **AI/ML**: Google's Gemini-2.0-flash model 
- **Vector Database**: Pinecone for similarity search
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **PDF Processing**: PyPDF2

## Setup Instructions

### Installation
1. Clone the repository:
git clone https://github.com/Vedant988/ResuMatch.git
cd resumatch
Copy
2. Install dependencies:
pip install -r requirements.txt
Copy
3. Create a .env file in the project root directory with the following variables:
   
GOOGLE_API_KEY=your_google_api_key, 
PINECONE_API_KEY=your_pinecone_api_key, 
PINECONE_ENVIRONMENT=your_pinecone_environment 

5. Run the application:
python app.py
Copy
6. Open your browser and go to:
http://localhost:5000
Copy
## Getting API Keys

### Google API Key
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gemini API
4. Create credentials to obtain your API key

### Pinecone API Key
1. Sign up for a free account at [Pinecone](https://www.pinecone.io/)
2. Create a new project
3. Generate an API key from your project dashboard
4. Note your environment (e.g., us-east-1)

## Usage
1. Upload multiple resume PDFs and enter a job description to rank candidates
2. Or use the comparison feature to analyze two candidates side-by-side
3. View interactive visualizations of candidate strengths and weaknesses

### NOTE
You need to clear the vector database for each resume set after each trial, as the code works with this logic only. Please ensure to clear the database after each set of resume uploads or each run.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
