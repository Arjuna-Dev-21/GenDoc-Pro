# GenDoc-Pro 🤖📄

**An AI-Powered Code Documentation & Visualization Tool**

GenDoc-Pro is a Generative AI application designed to help developers instantly understand unfamiliar code. It automatically generates comprehensive documentation, visual flowcharts, and provides an interactive Q&A chat interface for source code files.

## 🚀 Features

*   **Automated Documentation:** Instantly generates high-level summaries and detailed function breakdowns.
*   **Visual Flowcharts:** Converts code logic into clear, visual diagrams using Mermaid.js.
*   **Interactive Chat (RAG):** Ask questions about your code (e.g., "What does this specific function do?") and get accurate answers based on the code context.
*   **PDF Export:** Download a professional PDF report containing the documentation and the flowchart.

## 🛠️ Tech Stack

*   **Python** (Core Logic)
*   **Streamlit** (User Interface)
*   **OpenAI GPT-4o-mini** (LLM for generation and Q&A)
*   **LangChain & FAISS** (RAG Architecture for the Chatbot)
*   **FPDF2** (PDF Generation)
*   **Mermaid.js** (Diagram Rendering)

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/GenDoc-Pro.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your API Key:
    *   Create a `.env` file in the root directory.
    *   Add your key: `OPENAI_API_KEY="sk-..."`
4.  Run the application:
    ```bash
    streamlit run app.py
    ```

## 📝 Capstone Project
Submitted as the final project for the Generative AI Bootcamp.