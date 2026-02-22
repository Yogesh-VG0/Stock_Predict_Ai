# integrating-redis.md 

So you just spun up a free 30MB Redis Cloud database (`redis-15118.crce176.me-central-1-1.ec2.cloud.redislabs.com:15118`) and want to use it to learn how Redis works in a real production app without risking breaking the site. 

**I have fantastic news:** Your Python Machine Learning Backend (`ml_backend/api/main.py`) already has a complete, production-ready Redis integration coded into it that fails gracefully if the database disconnects. It is currently dormant because it lacks the connection string. 

## Where Redis fits into your Architecture

If you inject your Redis URL into the Python backend, it will automatically enable two highly educational backend patterns:

### 1. Sliding Window Rate Limiting (`ml_backend/api/rate_limiter.py`)
Currently, your ML API has no protection against spam requests. If enabled, the `RateLimiterMiddleware` will use Redis to track how many requests an IP address makes. 
- It uses the `INCR` command to atomically count requests per IP.
- It uses the `EXPIRE` command to automatically reset the counter every hour.
- If a user exceeds 100 requests/hour, they get an HTTP 429 response.

### 2. Time-To-Live (TTL) Response Caching (`ml_backend/api/cache.py`)
When a user requests predictions for `AAPL`, the Python backend normally has to query MongoDB, deserialize the data, and format the JSON.
- With Redis enabled, when `AAPL` is queried, the entire JSON response is serialized and stored in Redis using `SETEX` for 60 seconds.
- For the next 60 seconds, any user requesting `AAPL` receives the raw string directly from Redis RAM (sub-millisecond latency) bypassing MongoDB entirely!

## How to Implement It Safely

Because the Python backend uses a `try/except` block and sets `redis_client = None` upon failure, **this is 100% safe.** If your free Redis tier hits its connection limit or goes to sleep, your backend will simply fall back to querying MongoDB. 

### Step 1: Format your URL
Your connection string is:
`redis://default:*******@redis-15118.crce176.me-central-1-1.ec2.cloud.redislabs.com:15118`

> [!NOTE]
> Make sure to replace `*******` with your actual Redis password. Since you are using a free tier without TLS enabled, ensure you **do not** use `rediss://` (with two S's). Keep it as `redis://`.

### Step 2: Inject the Environment Variable
You need to add this as an Environment Variable to wherever you are hosting the FastAPI backend (e.g., Koyeb, Render, or Railway):

**Key:** `REDIS_URL`
**Value:** `redis://default:YOUR_PASSWORD@redis-15118.crce176...`

### Step 3: Verify it works!
1. Restart your Python backend.
2. In the deployment logs, you should see the backend successfully connect instead of skipping the cache.
3. Open your terminal and connect using the `redis-cli` command you copied from Redis Cloud.
4. Run the command `KEYS *`. You should start seeing keys look like:
   - `ratelimit:192.168.1.1`
   - `predictions:v1:AAPL`

This is the perfect, risk-free way to learn how caching layers and rate-limiters operate in a distributed system!
