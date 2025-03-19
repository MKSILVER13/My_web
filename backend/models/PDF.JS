const mongoose = require('mongoose');

const GraphSchema = new mongoose.Schema({
    filename: { type: String, required: true },
    graphData: { type: Object, required: true }, // Contains nodes and edges
    htmlOutputPath: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Graph', GraphSchema);
