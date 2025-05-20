require('dotenv').config(); // Moved to the top
const express = require('express');
const mongoose = require('mongoose'); // Kept for now, as uploadRoute might use Mongoose schemas
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const uploadRoute = require('./routes/upload');

const app = express();
const PORT = process.env.PORT || 3000;

// New MongoDB Atlas connection logic (for MongoClient, primarily for ping test here)
const { MongoClient, ServerApiVersion } = require('mongodb');
const fullAtlasUri = process.env.MONGO_ATLAS_URI;

if (!fullAtlasUri) {
    console.error("Error: MONGO_ATLAS_URI not found in .env file. Please ensure it is set.");
    process.exit(1);
}

// For MongoClient ping, use a URI without a specific database name initially.
// The ping command itself will target the 'admin' database.
// const pingClientUri = fullAtlasUri.substring(0, fullAtlasUri.lastIndexOf('/')) + '/?retryWrites=true&w=majority&appName=Cluster0';

const mongoPingClient = new MongoClient(fullAtlasUri, { // Use fullAtlasUri directly for the ping client as well
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  }
});

async function connectToAtlasAndPing() {
  try {
    await mongoPingClient.connect();
    await mongoPingClient.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment (via MongoClient). You successfully connected to MongoDB Atlas!");
  } catch (err) {
    console.error("Failed to connect and ping MongoDB Atlas (via MongoClient):", err);
    // We might not want to exit here if Mongoose connection is the primary one for operations
  } finally {
    await mongoPingClient.close();
  }
}

// Mongoose connection URI (this will be used by your models)
const mongooseAtlasUri = process.env.MONGO_ATLAS_URI; // Mongoose uses the full URI from .env

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// Middleware
app.use(cors({
    origin: '*',  // Allow all origins for development
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from specific directories
app.use('/frontend', express.static(path.join(__dirname, '..', 'frontend')));
app.use('/uploads', express.static(uploadsDir));
app.use('/lib', express.static(path.join(__dirname, '..', 'lib')));

// Redirect root to frontend
app.get('/', (req, res) => {
    res.redirect('/frontend');
});

// API Routes
app.use('/upload', uploadRoute);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err.stack);
    res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

// Connect Mongoose and start server
async function startServer() {
    try {
        // First, do the MongoClient ping test (optional, but good for diagnostics)
        await connectToAtlasAndPing();

        // Now, connect Mongoose
        await mongoose.connect(mongooseAtlasUri, { // Mongoose uses the full URI from .env
          // dbName: 'pdf_knowledge_graph', // This is part of the MONGO_ATLAS_URI
          serverApi: { // Ensure serverApi is configured for Mongoose as well
            version: ServerApiVersion.v1,
            strict: true,
            deprecationErrors: true
          }
          // useNewUrlParser and useUnifiedTopology are deprecated and removed
        });
        console.log('Mongoose connected to MongoDB Atlas (pdf_knowledge_graph database)');

        // Start server only after Mongoose is connected
        app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`);
            console.log(`Frontend available at http://localhost:${PORT}/frontend`);
        });

    } catch (err) {
        console.error('Failed to connect to MongoDB via Mongoose or start server:', err);
        process.exit(1); // Exit if Mongoose connection fails
    }
}

startServer();