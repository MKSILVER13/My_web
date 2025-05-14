const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const Graph = require('../models/PDF'); // MongoDB model

const router = express.Router();

// Configure storage for uploaded files
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, path.join(__dirname, '../uploads/'));
  },
  filename: function(req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

router.post('/', upload.single('pdf-file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    const filePath = req.file.path;
    const fileName = req.file.originalname; // if you still want to store the original name
    const uniqueName = req.file.filename;   // use the unique generated name from multer
    const baseName = path.basename(uniqueName, path.extname(uniqueName)); // ensure uniqueness in output paths
    
    
    // Define output paths using absolute paths to avoid path resolution issues
    const uploadsDir = path.join(__dirname, '../uploads');
    const jsonOutputPath = path.join(uploadsDir, `${baseName}_output.json`);
    const htmlOutputPath = path.join(uploadsDir, `${baseName}_output.html`);
    
    // Get path to the Python script (in parent directory)
    const pythonScriptPath = path.join(__dirname, '../../scripts/pdf_to_tree_final.py');
    
    console.log(`Processing PDF file: ${filePath}`);
    console.log(`Python script path: ${pythonScriptPath}`);
    console.log(`JSON output path: ${jsonOutputPath}`);
    console.log(`HTML output path: ${htmlOutputPath}`);
    
    // Execute the Python script with proper path resolution
    const command = `python "${pythonScriptPath}" "${filePath}" "${htmlOutputPath}" "${jsonOutputPath}"`;
    console.log(`Executing command: ${command}`);
    
    exec(command, async (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing Python script: ${error.message}`);
        console.error(`Python stderr: ${stderr}`);
        return res.status(500).json({ success: false, message: 'Error processing the PDF file' });
      }
      
      console.log(`Python script stdout: ${stdout}`);
      if (stderr) {
        console.log(`Python script stderr: ${stderr}`);
      }

      try {
        // Check if the JSON file exists
        if (!fs.existsSync(jsonOutputPath)) {
          console.error(`JSON output file not found at: ${jsonOutputPath}`);
          return res.status(500).json({ success: false, message: 'JSON output file not created' });
        }

        // Read and parse the graph data
        const graphData = JSON.parse(fs.readFileSync(jsonOutputPath, 'utf8'));
        
        // Format nodes for vis.js if needed
        const formattedGraphData = {
          nodes: graphData.nodes.map(node => ({
            id: node.id,
            label: node.id.substring(0, 30) + (node.id.length > 30 ? '...' : ''),
            title: node.content || 'No content',
            level: node.level,
            color: node.is_underlined ? '#FFD700' : getColorByLevel(node.level)
          })),
          edges: graphData.edges
        };
        
        // Save graph data to MongoDB
        const newGraph = new Graph({
          filename: fileName,
          graphData: formattedGraphData,
          htmlOutputPath: htmlOutputPath
        });

        const savedGraph = await newGraph.save();
        
        res.json({
          success: true,
          message: 'PDF processed and graph saved successfully',
          graphData: formattedGraphData,
          id: savedGraph._id
        });
      } catch (jsonError) {
        console.error(`Error processing JSON data: ${jsonError.message}`);
        res.status(500).json({ success: false, message: `Error processing JSON data: ${jsonError.message}` });
      }
    });
  } catch (err) {
    console.error(`Unexpected error in upload route: ${err.message}`);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Get color based on node level
function getColorByLevel(level) {
  const colors = [
    '#ff0000',  // Level 0 - Red
    '#00cc00',  // Level 1 - Green
    '#0099ff',  // Level 2 - Blue
    '#9933ff',  // Level 3 - Purple
    '#ff9900'   // Level 4 - Orange
  ];
  return colors[level] || "#cccccc";
}

// Route to get a specific graph by ID
router.get('/:id', async (req, res) => {
  try {
    const graph = await Graph.findById(req.params.id);
    if (!graph) {
      return res.status(404).json({ success: false, message: 'Graph not found' });
    }
    res.json({ success: true, graphData: graph.graphData });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;
