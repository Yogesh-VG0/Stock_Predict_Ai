const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const newsRoutes = require('./routes/newsRoutes');
const marketRoutes = require('./routes/market');

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

app.use('/api/news', newsRoutes);
app.use('/api/market', marketRoutes);

module.exports = app; 