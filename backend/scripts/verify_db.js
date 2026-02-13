const mongoose = require('mongoose');

// Use standard connection string but add directConnection=true to try and force it
// effectively treating it as a standalone for a moment to just get data
const MONGODB_URI = 'mongodb://yogesh:StockProject45@ac-m8tzhmj-shard-00-00.m8tzhmj.mongodb.net:27017/stock_predictor?ssl=true&authSource=admin&retryWrites=true&w=majority&directConnection=true';

async function verifyData() {
    try {
        console.log('🔌 Connecting to MongoDB (Direct Connection)...');

        // Set strictQuery to suppress warning (and maybe help?)
        mongoose.set('strictQuery', false);

        await mongoose.connect(MONGODB_URI, {
            serverSelectionTimeoutMS: 5000, // Fail faster
            connectTimeoutMS: 10000,
        });
        console.log('✅ Connected!');

        // Access native driver for direct collection access
        const db = mongoose.connection.db;
        const ticker = 'AAPL';
        console.log(`\n🔍 Checking data for ${ticker}...`);

        // 1. Check Stock Predictions
        const predictionsCol = db.collection('stock_predictions');

        const count = await predictionsCol.countDocuments({ ticker: ticker });
        console.log(`   (Raw count in 'stock_predictions' for ${ticker}: ${count})`);

        if (count > 0) {
            // Get all docs for this ticker to see structure
            const docs = await predictionsCol.find({ ticker: ticker }).sort({ timestamp: -1 }).limit(5).toArray();

            console.log(`   Found ${docs.length} recent docs.`);
            docs.forEach(d => {
                console.log(`   ---`);
                console.log(`   ID: ${d._id}`);
                console.log(`   Timestamp: ${d.timestamp}`);
                console.log(`   Window: ${d.window}`);
                console.log(`   Predicted Price: ${d.predicted_price}`);
            });
        } else {
            console.log('   ❌ No documents found for AAPL');
        }

        // 2. Check Explanations
        const explanationCol = db.collection('prediction_explanations');
        const expCount = await explanationCol.countDocuments({ ticker: ticker });
        console.log(`\n   (Raw count in 'prediction_explanations' for ${ticker}: ${expCount})`);

    } catch (error) {
        console.error('❌ Error:', error.message);
        if (error.cause) console.error('   Cause:', error.cause);
    } finally {
        if (mongoose.connection) await mongoose.connection.close();
    }
}

verifyData();
