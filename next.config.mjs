/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  output: 'standalone', // Enable standalone output for Docker deployment
  outputFileTracingRoot: process.cwd(),
  async rewrites() {
    // Use environment variables for production deployments
    const backendUrl = process.env.NODE_BACKEND_URL || process.env.NEXT_PUBLIC_NODE_BACKEND_URL || 'http://localhost:5000';
    
    return [
      {
        source: '/api/news/:path*',
        destination: `${backendUrl}/api/news/:path*`,
      },
      {
        source: '/api/stock/:path*',
        destination: `${backendUrl}/api/stock/:path*`,
      },
      {
        source: '/api/watchlist/:path*',
        destination: `${backendUrl}/api/watchlist/:path*`,
      },
      {
        source: '/api/market/:path*',
        destination: `${backendUrl}/api/market/:path*`,
      },
    ];
  },
}

export default nextConfig
