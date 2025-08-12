# FOSS Agentic Workshop IGI

This repository contains materials and code from a 2-day workshop conducted at FOSS Mumbai. The workshop introduced participants to the fundamentals of prompting, Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and agent-based systems.

## Workshop Overview

- **Event:** FOSS Mumbai Workshop
- **Duration:** 2 Days
- **Topics Covered:**
  - Basics of Prompt Engineering
  - Introduction to NLP concepts
  - Retrieval-Augmented Generation (RAG)
  - Building and working with AI Agents

## Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/Foss-agentic-workshop-igi.git
   cd Foss-agentic-workshop-igi
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Project

1. **Start the Prompt Application**
   ```sh
   streamlit run prompt_bot.py
   ```
2. **Populate chromadb**
```sh
python populate_db.py
```
3. **Start the RAG Application**
   ```sh
   streamlit run rag_bot.py
   ```


## Additional Resources

- Workshop slides and notebooks are available in the repository.
- For questions or feedback, please open an issue or contact the organizers.

---
Happy Learning!
