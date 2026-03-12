# LLM-TicketRouter

An end-to-end Streamlit workspace that uses a Pinecone-backed vector store, OpenAI question-answering, and a support-vector-machine classifier to route incoming employee tickets into HR, IT, or Transportation queues. The repository supplies both the customer-facing help desk (`app.py`) and the internal utilities needed to retrain the classifier, refresh embeddings, and track pending tickets.

## Key components
- `app.py` – Streamlit front end that gathers user questions, retrieves context from the Pinecone index, answers the user with LangChain/OpenAI, and (if requested) predicts ticket routing with the pretrained `modelsvm.pk1`.
- `user_utils.py` – Embedding helpers (`HuggingFaceEmbeddings`), Pinecone index loading, document similarity search, LangChain QA chain wiring, and the SVM prediction wrapper.
- `pages/Create_ML_Model.py` – Admin UI for loading `Tickets.csv`, creating embeddings in `all-MiniLM-L6-v2`, training/evaluating the `SVC` pipeline, and rebuilding `modelsvm.pk1`.
- `pages/Load_Data_Store.py` – Upload PDFs, split them into chunks, embed with Hugging Face, and push them into the Pinecone `tickets` index (requires a valid Pinecone key/environment).
- `pages/Pending_tickets.py` – Simple viewer for the session-state-backed queues that `app.py` populates once a ticket is submitted.
- `pages/admin_utils.py` – Data-processing helpers shared by the admin pages (PDF reading, chunking, dataset splitting, scoring).
- `Tickets.csv` – Sample dataset (text,label) for training new classifiers; it covers HR, IT, and Transportation ticket scenarios.

## Getting started
1. **Install dependencies.**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment variables.** Create a `.env` (or update the shipped `.env.example`) with `OPENAI_API_KEY`. The Pinecone helpers currently set `PINECONE_API_KEY` inside the admin pages; replace the hard-coded placeholder with your real key or update the files to read from the environment. You may also define:
   - `PINECONE_API_KEY` (required for index pushes/reads)
   - `PINECONE_ENVIRONMENT` (defaults to `us-east-1` inside the code)
   - `PINECONE_INDEX_NAME` (the code uses `tickets`)

## Running the customer-facing app
```bash
streamlit run app.py
```
The interface prompts the user for their issue, retrieves the most relevant chunks from Pinecone, shows the LangChain/OpenAI-generated answer, and offers a “Submit ticket?” button that embeds the input, predicts the department, and stores it in the session-state queue you can review in `pages/Pending_tickets.py`.

## Training or refreshing the SVM classifier
1. `streamlit run pages/Create_ML_Model.py`
2. Upload a CSV that mirrors `Tickets.csv` (two columns: message text, department label). The training tab splits the data, trains a `StandardScaler + SVC(class_weight='balanced')` pipeline on `all-MiniLM-L6-v2` embeddings, and stores it in `st.session_state`.
3. Use the evaluation tab to check accuracy and run sample queries.
4. Save the model (generates `modelsvm.pk1`) from the fourth tab. The main app will then use the updated classifier.

## Maintaining the Pinecone vector store
1. `streamlit run pages/Load_Data_Store.py`
2. Upload PDFs you want searchable.
3. The page splits each document, creates embeddings, and pushes them into the `tickets` index. Verify that `PINECONE_API_KEY` is set before running or replace the temporary assignment with `os.getenv`.

## Viewing pending tickets
```bash
streamlit run pages/Pending_tickets.py
```
Each tab lists the session-state tickets that `app.py` has routed to HR, IT, or Transportation.

## Data
- **`Tickets.csv`** – The training CSV is comma-delimited, without headers. Column 1 is the ticket text and column 2 is the label (`HR`, `IT`, or `Transportation`). Use this as a template when collecting new training data.
- **`modelsvm.pk1`** – Pretrained SVM pipeline (scikit-learn + `joblib`) that classifies combined embeddings for ticket routing.

## Tips & next steps
- Consider replacing the hard-coded Pinecone keys with proper environment-variable reads before sharing or deploying the repo.
- Extend `pages/Create_ML_Model.py` with automated tracking of evaluation metrics or new label categories if your organization grows beyond the current three departments.
- After reindexing documents, restart the main app so it reads the refreshed Pinecone contents.
