# app.py
import streamlit as st
import os
import base64
import requests
import tempfile
from dotenv import load_dotenv
import openai
import json
from fpdf import FPDF
from streamlit_mermaid import st_mermaid

# --- LangChain Imports ---
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import CharacterTextSplitter
except ImportError as e:
    st.error(f"Import Error: {e}. Please run: pip install langchain-openai langchain-community faiss-cpu")
    st.stop()

# --- NEW: AI Core Logic ---
def get_prompt(code_content, language):
    """
    Creates a detailed, structured prompt for the LLM.
    """
    prompt = f"""
    You are an expert software developer and technical writer. Your task is to analyze the provided source code and generate comprehensive documentation.

    **Instructions:**
    1.  Analyze the following `{language}` code.
    2.  Provide a high-level summary.
    3.  Break down each function/class (Parameters, Returns, Purpose).
    4.  Generate a flowchart using **Mermaid.js `graph TD` syntax**.
        - **IMPORTANT:** Output ONLY the code for the flowchart inside the flowchart section. Do not wrap it in markdown code blocks (like ```mermaid). Just raw text.
        - Keep the flowchart logic simple and clear.
    5.  The entire output must be in a single Markdown format.

    **Output Structure:**
    Use the following headers for each section:

    ### Summary

    ### Breakdown

    ### Flowchart

    ---
    **Code to Analyze:**
    ```
    {code_content}
    ```
    ---
    """
    return prompt

def generate_documentation(api_key, code_content, language):
    """
    Calls the OpenAI API to generate documentation.
    """
    try:
        # Set the API key for the openai library
        openai.api_key = api_key
        
        prompt = get_prompt(code_content, language)
        
        # Using the ChatCompletion endpoint
        response = openai.chat.completions.create(
            #model="gpt-4-turbo-preview",  # Or "gpt-3.5-turbo" for a faster, cheaper option
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic, factual output
        )
        
        # Extract the content from the response
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# --- PDF Generation Logic ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'GenDoc-Pro Documentation', new_x="LMARGIN", new_y="NEXT", align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        # Fixed the DeprecationWarning here
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_pdf(summary, breakdown, flowchart_code):
    pdf = PDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Code Documentation", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(5)
    
    # 1. Summary Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "1. Summary", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.set_font("Helvetica", '', 11)
    safe_summary = summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, safe_summary)
    pdf.ln(5)
    
    # 2. Breakdown Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "2. Detailed Breakdown", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.set_font("Helvetica", '', 11)
    clean_breakdown = breakdown.replace("**", "").encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, clean_breakdown)
    pdf.ln(5)
    
    # 3. Flowchart Image
    if flowchart_code:
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, "3. Process Flowchart", new_x="LMARGIN", new_y="NEXT", align='L')
        
        try:
            # 1. Wrap code in a JSON object with theme config
            mermaid_data = {
                "code": flowchart_code,
                "mermaid": {"theme": "default"}
            }
            json_str = json.dumps(mermaid_data)
            
            # 2. Use URL-SAFE Base64 encoding (replaces + and / with - and _)
            json_bytes = json_str.encode("utf8")
            base64_bytes = base64.urlsafe_b64encode(json_bytes)
            base64_string = base64_bytes.decode("ascii")
            
            # 3. Construct URL (No ?bgColor needed anymore)
            image_url = f"https://mermaid.ink/img/{base64_string}"

            response = requests.get(image_url)
            
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                pdf.image(tmp_path, x=10, w=190)
            else:
                pdf.set_font("Helvetica", 'I', 10)
                pdf.cell(0, 10, f"(Flowchart server error: {response.status_code})", new_x="LMARGIN", new_y="NEXT")
                
        except Exception as e:
            print(f"DEBUG: Exception: {e}")
            pdf.set_font("Helvetica", 'I', 10)
            pdf.cell(0, 10, f"(Error rendering flowchart: {e})", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

# --- Q&A Logic (RAG) ---
def build_vector_store(text_content):
    """
    Splits the code into chunks and builds a searchable vector store.
    """
    # 1. Split text into chunks (so we don't exceed token limits)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)
    
    # 2. Create Embeddings (turn text into numbers)
    # We use OpenAI's embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # 3. Create Vector Store (FAISS)
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# --- Page Configuration ---
st.set_page_config(
    page_title="GenDoc-Pro",
    page_icon="🤖",
    layout="wide"
)

# --- App Title and Description ---
st.title("📄 GenDoc-Pro: AI Code Documenter")
st.write("Upload your source code file and let AI generate the documentation, flowcharts, and answer your questions.")

# --- API Key Management (Sidebar) ---
st.sidebar.header("Configuration")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 

if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your API key to proceed.")
        st.stop()
else:
    st.sidebar.success("API Key loaded successfully.")

# --- Core Application ---
# Dynamically get file extension for language parameter
file_extension_map = {
    'py': 'Python', 'js': 'JavaScript', 'php': 'PHP', 
    'java': 'Java', 'cs': 'C#', 'cpp': 'C++'
}

uploaded_file = st.file_uploader(
    "Choose a source code file", 
    type=list(file_extension_map.keys())
)

if uploaded_file is not None:
    # Clear previous vector store if a new file is uploaded
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        if 'vector_store' in st.session_state:
            del st.session_state.vector_store
        st.session_state.last_uploaded = uploaded_file.name
        
    code_content = uploaded_file.getvalue().decode("utf-8")
    file_extension = uploaded_file.name.split('.')[-1]
    language = file_extension_map.get(file_extension, "plaintext")
    
    st.subheader(f"Your {language} Code:")
    st.code(code_content, language=language.lower())

    if st.button("✨ Generate Documentation"):
        if not api_key:
            st.error("API Key is not set. Please enter it in the sidebar.")
        else:
            with st.spinner("Generating documentation... This may take a moment."):
                # --- THIS IS THE NEW, REAL AI CALL ---
                generated_content = generate_documentation(api_key, code_content, language)
                
                # --- NEW: Parsing Logic ---
                # We split the content based on our structured prompt headers
                try:
                    # 1. Split into Summary and the rest
                    parts = generated_content.split("### Breakdown")
                    
                    if len(parts) > 0:
                        st.session_state.summary = parts[0].replace("### Summary", "").strip()
                    else:
                        st.session_state.summary = "Summary not found."

                    if len(parts) > 1:
                        # 2. Split the rest into Breakdown and Flowchart
                        rest = parts[1]
                        breakdown_parts = rest.split("### Flowchart")
                        
                        st.session_state.breakdown = breakdown_parts[0].strip()
                        
                        if len(breakdown_parts) > 1:
                            # 3. Clean up the flowchart code
                            raw_flowchart = breakdown_parts[1].strip()
                            
                            # Clean up markdown fences just in case the AI added them despite instructions
                            clean_flowchart = raw_flowchart.replace("```mermaid", "").replace("```", "").strip()
                            
                            st.session_state.flowchart_code = clean_flowchart
                        else:
                            st.session_state.flowchart_code = "Flowchart not found in response."
                    else:
                        st.session_state.breakdown = "Breakdown not found."
                        st.session_state.flowchart_code = ""

                    st.success("Documentation generated successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred during parsing: {e}")

# --- Display Results ---
if 'summary' in st.session_state:
    st.subheader("💬 Generated Documentation")
    
    with st.container(border=True):
        st.markdown("### Summary")
        st.markdown(st.session_state.summary)

    with st.container(border=True):
        st.markdown("### Breakdown")
        st.markdown(st.session_state.breakdown)

if 'flowchart_code' in st.session_state:
    st.subheader("📊 Flowchart")
    with st.container(border=True):
        # Render the diagram visually
        st_mermaid(st.session_state.flowchart_code, height="500px")
        
        # Add this expandable box so we can check the code if the image fails
        with st.expander("See Raw Mermaid Code (For Debugging)"):
            st.text(st.session_state.flowchart_code)

# --- Place this AFTER the Flowchart display block ---

if 'summary' in st.session_state and 'breakdown' in st.session_state:
    st.divider()
    st.subheader("📥 Export")
    
    if st.button("Generate PDF Report"):
        with st.spinner("Compiling PDF..."):
            pdf_data = create_pdf(
                st.session_state.summary, 
                st.session_state.breakdown, 
                st.session_state.flowchart_code
            )
            
            st.download_button(
                label="Download Documentation (PDF)",
                data=pdf_data,
                file_name="GenDoc_Report.pdf",
                mime="application/pdf"
            )

if 'summary' in st.session_state:
    st.divider()
    st.subheader("💬 Chat with your Code")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Handle User Input
    if prompt := st.chat_input("Ask a question about your code..."):
        
        # 1. Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check/Build Vector Store
                    if 'vector_store' not in st.session_state:
                         st.session_state.vector_store = build_vector_store(code_content)

                    # A. Search for relevant chunks (The "Retrieval" part)
                    docs = st.session_state.vector_store.similarity_search(prompt)
                    
                    # B. Prepare the Context (Combine the found code chunks)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # C. Send to OpenAI (The "Generation" part)
                    # We manually craft the prompt here.
                    rag_prompt = f"""
                    You are a helpful coding assistant. Answer the question based ONLY on the provided code context.
                    
                    Question: {prompt}
                    
                    Context (Relevant Code):
                    {context_text}
                    """
                    
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": rag_prompt}
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")