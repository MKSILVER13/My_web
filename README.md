# Knowledge Graph and Attendance Management System

This project is a web-based application designed to generate knowledge graphs from PDF files and manage attendance for club members. It includes a backend for processing PDFs and storing data, as well as a frontend for user interaction and visualization.

## Project Structure

```
my_web/
├── attendance.html          # Attendance management page
├── iit.json                 # Sample JSON data
├── iitg_knowledge_graph.html # Example knowledge graph visualization
├── iitg.pdf                 # Sample PDF file
├── index.html               # Homepage
├── news.html                # News updates page
├── pdf_to.py                # Python script for PDF processing and knowledge graph generation
├── projects.html            # Projects page with PDF upload functionality
├── styles.css               # Global styles for the frontend
├── backend/                 # Backend folder
│   ├── app.js               # Main backend application file
│   ├── package.json         # Node.js dependencies
│   ├── models/              # Mongoose models
│   │   └── PDF.JS           # Schema for storing graph data
│   ├── routes/              # API routes
│   │   └── upload.js        # Route for handling PDF uploads
│   ├── uploads/             # Directory for uploaded and processed files
│   └── lib/                 # Libraries for backend
├── frontend/                # Frontend folder
│   ├── attendance.html      # Attendance management page
│   ├── index.html           # Homepage
│   ├── news.html            # News updates page
│   ├── projects.html        # Projects page with PDF upload functionality
│   ├── styles.css           # Global styles for the frontend
├── lib/                     # Frontend libraries
│   ├── vis-9.1.2/           # Vis.js library for graph visualization
│   ├── tom-select/          # Additional frontend libraries
```

## Features

### Frontend
- **Homepage**: Provides an overview of the application.
- **Projects Page**: Allows users to upload PDF files and visualize knowledge graphs.
- **News Page**: Displays the latest news updates.
- **Attendance Page**: Manages attendance for club members.

### Backend
- **PDF Processing**: Uses `pdf_to.py` to extract structured data from PDFs and generate knowledge graphs.
- **Data Storage**: Stores graph data in MongoDB using Mongoose.
- **File Management**: Handles file uploads and stores processed outputs in the `uploads/` directory.

## Installation

### Prerequisites
- Node.js
- Python 3
- MongoDB

### Backend Setup
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the backend server:
   ```bash
   node app.js
   ```

### Frontend Setup
1. Open the `frontend/index.html` file in your browser.

### Python Script Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the `pdf_to.py` script manually if needed:
   ```bash
   python pdf_to.py <input_pdf> <output_html> <output_json>
   ```

## Usage

1. Open the `frontend/projects.html` file in your browser.
2. Upload a PDF file to generate a knowledge graph.
3. View the graph visualization and interact with nodes to see details.
4. Use the "Upload Another PDF" button to reset the interface and upload a new file.

## Libraries Used

### Frontend
- [Vis.js](https://visjs.github.io/vis-network/) for graph visualization
- Custom CSS for styling

### Backend
- [Express.js](https://expressjs.com/) for server-side routing
- [Mongoose](https://mongoosejs.com/) for MongoDB integration
- [Multer](https://github.com/expressjs/multer) for file uploads

### Python
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
- [NetworkX](https://networkx.org/) for graph data structure
- [PyVis](https://pyvis.readthedocs.io/) for graph visualization

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.