# PhilanthroBot

PhilanthroBot: A Trust-Centric Conversational Agent for NGO Discovery and Recommendation using Stateful RAG Architecture

## Overview

PhilanthroBot is an intelligent conversational agent designed to help users discover and recommend NGOs based on trust-centric criteria. Leveraging a stateful Retrieval-Augmented Generation (RAG) architecture, PhilanthroBot provides reliable, context-aware recommendations and information about various NGOs.

## Features

- Conversational interface for NGO discovery
- Trust-centric recommendation system
- Stateful RAG architecture for enhanced context and accuracy
- Integration with a local ChromaDB for fast retrieval
- Support for PDF-based NGO profiles

## Project Structure

```
app.py
requirements.txt
chroma_db/
	chroma.sqlite3
	...
ngo_profiles/
	1_akshaya_pratha_foundation.pdf
	2_CRY-child_rights_and_you.pdf
	...
```

## Getting Started

1. **Install dependencies:**

   ```powershell
   pip install -r .\requirements.txt
   ```

2. **Run the application:**
   ```powershell
   python .\app.py
   ```

## Data

- **NGO Profiles:** Located in the `ngo_profiles/` directory as PDF files.
- **Database:** Uses ChromaDB (`chroma_db/`) for fast vector-based retrieval.

## Requirements

See `requirements.txt` for Python package dependencies.

## License

Specify your license here.
