# TagChat-Library

TagChat-Library is an advanced AI-powered system that allows users to categorize and interact with uploaded PDFs dynamically. With the power of Google's Gemini LLM, users can label their resources with relevant tags, upload documents, and then chat directly with selected documents without processing the entire dataset. This project is particularly useful for students, researchers, and professionals managing large repositories of books, papers, and other resources.

## Features

1. **Tag-Based Organization**  
   - Users can create specific tags (e.g., "Physics," "Math," "History") to categorize their uploads.
   - Multiple tags can be assigned to a single document for better discoverability.

2. **Dynamic Document Selection**  
   - Instead of searching the entire library, users can narrow their focus by choosing relevant tags and selecting specific documents to chat with.

3. **Efficient PDF Interaction**  
   - No need to process the entire library—simply choose a tagged document, and the system will focus on the selected file.
   - Supports question-answering and summarization for specific documents.

4. **Powered by Gemini LLM**  
   - Provides state-of-the-art natural language understanding and responses for document interaction.

5. **Customizable and Scalable**  
   - Flexible architecture to accommodate different domains and file types.
   - Easily extendable for future features such as real-time updates or multi-user support.

## How It Works

1. **Add Tags**:  
   - Users create custom tags for their files to organize them effectively. For example, "Physics" for textbooks, "AI" for research papers, or "Finance" for reports.

2. **Upload Files**:  
   - Upload any PDF file under the selected tag(s).

3. **Chat with Files**:  
   - Select a tag and pick a document to start a conversation. Ask questions, request summaries, or extract specific information—all powered by Gemini LLM.

4. **Multi-Tag Search**:  
   - Use multiple tags to filter files and refine your selection for easier access to the required document.

## Tech Stack

- **Backend**: Python (FastAPI/Flask)
- **Frontend**: streamlit (Optional for GUI)
- **Database**: json for managing tags, user files, and metadata
- **File Storage**: Local File System
- **AI Model**: Google Gemini LLM via API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/unaib-tech/TagChat-Library.git
   cd TagChat-Library
