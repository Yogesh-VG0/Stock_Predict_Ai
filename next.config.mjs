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

  // Performance optimizations
  compiler: {
    removeConsole: false, // temporarily disabled for debugging
  },

  // Optimize package imports
  modularizeImports: {
    'lucide-react': {
      transform: 'lucide-react/dist/esm/icons/{{member}}',
    },
    'framer-motion': {
      transform: 'framer-motion/dist/es/{{member}}',
    },
  },

  // Experimental optimizations
  experimental: {
    optimizePackageImports: ['lucide-react', 'framer-motion', 'recharts', 'date-fns'],
  },

  async rewrites() {
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

  // Headers for caching static assets
  async headers() {
    return [
      {
        source: '/:all*(svg|jpg|png|webp|gif|ico)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },
}

export default nextConfig
