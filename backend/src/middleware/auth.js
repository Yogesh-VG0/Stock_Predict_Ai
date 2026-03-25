const jwt = require('jsonwebtoken');
const crypto = require('crypto');

const JWT_SECRET = process.env.JWT_SECRET_KEY || 'stockpredict-dev-secret-change-in-production';
const TOKEN_EXPIRY = '90d'; // Anonymous sessions last 90 days

/**
 * Issue a new anonymous session token.
 * The token contains a random userId (UUID-v4-style) so each browser
 * gets its own isolated watchlist without registration.
 */
function issueToken() {
  const userId = crypto.randomUUID();
  const token = jwt.sign({ sub: userId }, JWT_SECRET, { expiresIn: TOKEN_EXPIRY });
  return { token, userId };
}

/**
 * Verify and decode a Bearer token.
 * Returns the payload ({ sub: <userId>, iat, exp }) or null.
 */
function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch {
    return null;
  }
}

/**
 * Express middleware – requires a valid Bearer token on protected routes.
 * Attaches `req.userId` (string) extracted from the token's `sub` claim.
 * If no valid token is present, responds with 401.
 */
function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ success: false, error: 'Authentication required' });
  }

  const token = authHeader.slice(7); // strip "Bearer "
  const payload = verifyToken(token);
  if (!payload || !payload.sub) {
    return res.status(401).json({ success: false, error: 'Invalid or expired token' });
  }

  req.userId = payload.sub;
  next();
}

/**
 * POST /api/auth/session — issue (or refresh) an anonymous session.
 * The frontend calls this once on first visit, stores the token in
 * localStorage, and sends it as a Bearer header on subsequent requests.
 */
function sessionHandler(req, res) {
  // If the caller already has a valid token, return the same userId
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const payload = verifyToken(authHeader.slice(7));
    if (payload && payload.sub) {
      // Re-sign to extend expiry
      const token = jwt.sign({ sub: payload.sub }, JWT_SECRET, { expiresIn: TOKEN_EXPIRY });
      return res.json({ token, userId: payload.sub });
    }
  }

  // No valid token — create a new anonymous session
  const { token, userId } = issueToken();
  res.json({ token, userId });
}

module.exports = { requireAuth, sessionHandler, issueToken, verifyToken, JWT_SECRET };
