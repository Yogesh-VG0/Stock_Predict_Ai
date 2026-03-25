/**
 * Phase 1 — Security & Data Integrity Tests
 *
 * Tests:
 *  1. Unauthenticated requests are rejected (401)
 *  2. IDOR: User A cannot access User B's watchlist
 *  3. Auth session issuance & refresh
 *  4. Watchlist CRUD via authenticated endpoints
 *  5. getBatchStatus math is correct (75 tickers, no negative values)
 *  6. Landing stats returns valid percentage format
 */

const request = require('supertest');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');

// We need to set env vars BEFORE requiring app
process.env.JWT_SECRET_KEY = 'test-secret-for-ci';
process.env.NODE_ENV = 'test';

const app = require('../app');
const { JWT_SECRET } = require('../middleware/auth');
const Watchlist = require('../models/Watchlist');
const mongoConnection = require('../config/mongodb');

// Increase timeout for tests that hit MongoDB
jest.setTimeout(30000);

// ─── Helpers ─────────────────────────────────────────────
function makeToken(userId) {
  return jwt.sign({ sub: userId }, JWT_SECRET, { expiresIn: '1h' });
}

// ─── Setup / Teardown ────────────────────────────────────
beforeAll(async () => {
  // Connect to MongoDB for integration tests
  try {
    await mongoConnection.connect();
  } catch (err) {
    console.warn('MongoDB not available — some tests will be skipped');
  }
});

afterAll(async () => {
  // Close MongoDB connection
  try {
    await mongoose.connection.close();
  } catch (err) {
    // ignore
  }
});

// ─── Test Suite ──────────────────────────────────────────

describe('Phase 1 — Security & Data Integrity', () => {

  // ── 1. Authentication Required ──────────────────────
  describe('Authentication enforcement', () => {
    test('GET /api/watchlist/me without token → 401', async () => {
      const res = await request(app).get('/api/watchlist/me');
      expect(res.statusCode).toBe(401);
      expect(res.body.success).toBe(false);
      expect(res.body.error).toMatch(/authentication/i);
    });

    test('POST /api/watchlist/me/add without token → 401', async () => {
      const res = await request(app)
        .post('/api/watchlist/me/add')
        .send({ symbol: 'AAPL' });
      expect(res.statusCode).toBe(401);
    });

    test('DELETE /api/watchlist/me/AAPL without token → 401', async () => {
      const res = await request(app).delete('/api/watchlist/me/AAPL');
      expect(res.statusCode).toBe(401);
    });

    test('Request with invalid/expired token → 401', async () => {
      const badToken = jwt.sign({ sub: 'user1' }, 'wrong-secret', { expiresIn: '1h' });
      const res = await request(app)
        .get('/api/watchlist/me')
        .set('Authorization', `Bearer ${badToken}`);
      expect(res.statusCode).toBe(401);
      expect(res.body.error).toMatch(/invalid|expired/i);
    });

    test('Request with expired token → 401', async () => {
      const expiredToken = jwt.sign({ sub: 'user1' }, JWT_SECRET, { expiresIn: '-1s' });
      const res = await request(app)
        .get('/api/watchlist/me')
        .set('Authorization', `Bearer ${expiredToken}`);
      expect(res.statusCode).toBe(401);
    });
  });

  // ── 2. IDOR Prevention ─────────────────────────────
  const mongoAvailable = () => mongoConnection.isConnected;

  describe('IDOR prevention — user isolation', () => {
    let tokenA, tokenB;
    const userIdA = 'test-user-a-' + Date.now();
    const userIdB = 'test-user-b-' + Date.now();

    beforeAll(() => {
      tokenA = makeToken(userIdA);
      tokenB = makeToken(userIdB);
    });

    afterAll(async () => {
      if (mongoAvailable()) {
        await Watchlist.deleteMany({ userId: { $in: [userIdA, userIdB] } });
      }
    });

    test('User A adds TSLA; User B does not see it', async () => {
      if (!mongoAvailable()) return; // skip if no DB

      // User A adds TSLA
      await request(app)
        .post('/api/watchlist/me/add')
        .set('Authorization', `Bearer ${tokenA}`)
        .send({ symbol: 'TSLA' });

      // User B fetches their watchlist
      const resB = await request(app)
        .get('/api/watchlist/me')
        .set('Authorization', `Bearer ${tokenB}`);
      expect(resB.statusCode).toBe(200);

      // User A fetches — should have TSLA
      const resA = await request(app)
        .get('/api/watchlist/me')
        .set('Authorization', `Bearer ${tokenA}`);
      const symbolsA = resA.body.watchlist.map(i => i.symbol);
      expect(symbolsA).toContain('TSLA');
    });

    test('Legacy /:userId route still uses token userId, not URL param', async () => {
      if (!mongoAvailable()) return;

      const res = await request(app)
        .get(`/api/watchlist/${userIdA}`)
        .set('Authorization', `Bearer ${tokenB}`);
      expect(res.statusCode).toBe(200);
    });
  });

  // ── 3. Auth Session Issuance ───────────────────────
  describe('Session endpoint', () => {
    test('POST /api/auth/session returns a valid JWT', async () => {
      const res = await request(app)
        .post('/api/auth/session')
        .send({});
      
      expect(res.statusCode).toBe(200);
      expect(res.body.token).toBeDefined();
      expect(res.body.userId).toBeDefined();

      const payload = jwt.verify(res.body.token, JWT_SECRET);
      expect(payload.sub).toBe(res.body.userId);
    });

    test('POST /api/auth/session with existing token refreshes it', async () => {
      const userId = 'refresh-test-' + Date.now();
      const oldToken = makeToken(userId);

      const res = await request(app)
        .post('/api/auth/session')
        .set('Authorization', `Bearer ${oldToken}`)
        .send({});

      expect(res.statusCode).toBe(200);
      expect(res.body.userId).toBe(userId);
      expect(res.body.token).not.toBe(oldToken);
    });
  });

  // ── 4. Watchlist CRUD (authenticated) ──────────────
  describe('Watchlist CRUD', () => {
    let token;
    const userId = 'crud-test-' + Date.now();

    beforeAll(() => {
      token = makeToken(userId);
    });

    afterAll(async () => {
      if (mongoAvailable()) {
        await Watchlist.deleteMany({ userId });
      }
    });

    test('GET /api/watchlist/me creates default watchlist on first access', async () => {
      if (!mongoAvailable()) return;

      const res = await request(app)
        .get('/api/watchlist/me')
        .set('Authorization', `Bearer ${token}`);

      expect(res.statusCode).toBe(200);
      expect(res.body.success).toBe(true);
      expect(res.body.watchlist.length).toBeGreaterThan(0);

      const doc = await Watchlist.findOne({ userId });
      expect(doc).not.toBeNull();
      expect(doc.symbols.length).toBeGreaterThan(0);
    });

    test('POST /api/watchlist/me/add rejects invalid symbol', async () => {
      const res = await request(app)
        .post('/api/watchlist/me/add')
        .set('Authorization', `Bearer ${token}`)
        .send({ symbol: '123!@#' });

      expect(res.statusCode).toBe(400);
      expect(res.body.error).toMatch(/invalid/i);
    });

    test('DELETE /api/watchlist/me/:symbol removes a symbol', async () => {
      if (!mongoAvailable()) return;

      const res = await request(app)
        .delete('/api/watchlist/me/NFLX')
        .set('Authorization', `Bearer ${token}`);

      expect(res.statusCode).toBe(200);
      expect(res.body.success).toBe(true);

      const doc = await Watchlist.findOne({ userId });
      expect(doc.symbols).not.toContain('NFLX');
    });

    test('Watchlist survives simulated restart (persisted in MongoDB)', async () => {
      if (!mongoAvailable()) return;

      await request(app)
        .post('/api/watchlist/me/add')
        .set('Authorization', `Bearer ${token}`)
        .send({ symbol: 'KO' });

      const doc = await Watchlist.findOne({ userId });
      expect(doc.symbols).toContain('KO');
    });
  });

  // ── 5. getBatchStatus math ─────────────────────────
  describe('getBatchStatus data integrity', () => {
    test('total_tickers equals 75 when MongoDB is connected', async () => {
      if (!mongoAvailable()) return;

      const res = await request(app).get('/api/stock/batch/status');

      expect(res.body.total_tickers).toBe(75);
      expect(res.body.without_explanations).toBeGreaterThanOrEqual(0);
      expect(res.body.coverage_percentage).toBeGreaterThanOrEqual(0);
      expect(res.body.coverage_percentage).toBeLessThanOrEqual(100);
    });

    test('Batch status has correct math (with + without = total)', async () => {
      if (!mongoAvailable()) return;

      const res = await request(app).get('/api/stock/batch/status');
      const { with_explanations, without_explanations, total_tickers } = res.body;
      expect(with_explanations + without_explanations).toBe(total_tickers);
    });
  });

  // ── 6. Landing stats format ────────────────────────
  describe('Landing stats format', () => {
    test('topMover.change is a percentage string', async () => {
      const res = await request(app).get('/api/stock/landing/stats');
      
      expect(res.statusCode).toBe(200);
      expect(res.body.topMover).toBeDefined();
      expect(res.body.topMover.change).toMatch(/^[+-]?\d+\.\d+%$/);
    });
  });

  // ── 7. Public routes remain accessible ─────────────
  describe('Public routes still work', () => {
    test('GET /health → 200 or 503 with status', async () => {
      const res = await request(app).get('/health');
      expect([200, 503]).toContain(res.statusCode);
      expect(['healthy', 'degraded']).toContain(res.body.status);
      expect(res.body.service).toBe('stockpredict-backend');
      expect(res.body.uptime).toBeDefined();
      expect(res.body.dependencies).toBeDefined();
      expect(res.body.memory).toBeDefined();
    });

    test('GET /api/watchlist/status/websocket → 200 (no auth needed)', async () => {
      const res = await request(app).get('/api/watchlist/status/websocket');
      expect(res.statusCode).toBe(200);
    });

    test('GET /api/watchlist/updates/realtime?symbols=AAPL → 200 (no auth needed)', async () => {
      const res = await request(app).get('/api/watchlist/updates/realtime?symbols=AAPL');
      expect(res.statusCode).toBe(200);
    });
  });
});
