/**
 * Anonymous session management.
 * On first visit, calls POST /api/auth/session to get a JWT token.
 * Stores it in localStorage and returns it for Bearer auth headers.
 */

const SESSION_KEY = 'sp_session_token';

let tokenPromise: Promise<string | null> | null = null;

/** Get the base URL for the Node backend (same logic as api.ts) */
function getBackendUrl(): string {
  const isProduction = typeof window !== 'undefined' && window.location.hostname !== 'localhost';
  return isProduction ? '' : (process.env.NEXT_PUBLIC_NODE_BACKEND_URL || 'http://localhost:5000');
}

/**
 * Returns the current session token, requesting a new one if needed.
 * De-duplicates concurrent calls so only one network request is made.
 */
export async function getSessionToken(): Promise<string | null> {
  if (typeof window === 'undefined') return null; // SSR guard

  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;

  // Prevent multiple concurrent session requests
  if (!tokenPromise) {
    tokenPromise = requestNewSession();
  }

  const token = await tokenPromise;
  tokenPromise = null;
  return token;
}

async function requestNewSession(): Promise<string | null> {
  try {
    const res = await fetch(`${getBackendUrl()}/api/auth/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.token) {
      localStorage.setItem(SESSION_KEY, data.token);
      return data.token;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Build an Authorization header object for fetch calls.
 * Returns empty object if no token is available (graceful degradation).
 */
export async function authHeaders(): Promise<Record<string, string>> {
  const token = await getSessionToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

/**
 * Clear the stored session (e.g. for logout / reset).
 */
export function clearSession(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(SESSION_KEY);
  }
}
