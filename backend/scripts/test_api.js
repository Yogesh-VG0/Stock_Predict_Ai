const http = require('http');

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/api/stock/AAPL/predictions',
    method: 'GET',
};

const req = http.request(options, (res) => {
    console.log(`STATUS: ${res.statusCode}`);
    console.log(`HEADERS: ${JSON.stringify(res.headers)}`);

    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        console.log('BODY:', data);
        try {
            const json = JSON.parse(data);
            if (json.AAPL && json.AAPL.next_day && json.AAPL.next_day.predicted_price) {
                console.log(`✅ MongoDB Prediction Found: $${json.AAPL.next_day.predicted_price}`);
            } else {
                console.log('❌ Unexpected JSON structure or missing prediction');
            }
        } catch (e) {
            console.log('❌ Response is not valid JSON');
        }
    });
});

req.on('error', (e) => {
    console.error(`❌ Problem with request: ${e.message}`);
    console.error('   Is the backend server running on localhost:5000?');
});

req.end();
