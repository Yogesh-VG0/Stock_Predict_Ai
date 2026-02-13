const https = require('https');

const options = {
    hostname: 'present-mellie-stockpredict-986dc946.koyeb.app',
    port: 443,
    path: '/api/stock/AAPL/predictions',
    method: 'GET',
};

console.log(`🔍 Testing Live API: https://${options.hostname}${options.path}`);

const req = https.request(options, (res) => {
    console.log(`STATUS: ${res.statusCode}`);

    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        try {
            const json = JSON.parse(data);
            console.log('BODY:', JSON.stringify(json, null, 2));
            if (json.AAPL && json.AAPL['1_day'] && json.AAPL['1_day'].predicted_price) {
                console.log(`✅ Success! Real Prediction Found: $${json.AAPL['1_day'].predicted_price}`);
            } else {
                console.log('❌ Unexpected JSON structure or missing prediction');
            }
        } catch (e) {
            console.log('❌ Response is not valid JSON. First 100 chars:', data.substring(0, 100));
        }
    });
});

req.on('error', (e) => {
    console.error(`❌ Connection Problem: ${e.message}`);
});

req.end();
