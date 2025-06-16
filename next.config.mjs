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
  async rewrites() {
    return [
      {
        source: '/api/news/:path*',
        destination: 'http://localhost:5000/api/news/:path*',
      },
    ];
  },
}

export default nextConfig
