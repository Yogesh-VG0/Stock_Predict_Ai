#!/usr/bin/env node
/**
 * Generate favicon.ico, favicon-32x32.png, and apple-touch-icon.png from public/image.svg
 * Run: node scripts/generate-favicon.mjs   or   npm run generate-favicon
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import toIco from 'to-ico';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const svgPath = path.join(root, 'public', 'image.svg');
const publicDir = path.join(root, 'public');

if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });
if (!fs.existsSync(svgPath)) {
  console.error('Source SVG not found:', svgPath);
  process.exit(1);
}

const svgBuffer = fs.readFileSync(svgPath);

// Match image.svg dark background so favicon never shows white in search/browsers
const DARK_BG = { r: 13, g: 12, b: 14 };

async function main() {
  const resize32 = () => sharp(svgBuffer).resize(32, 32);
  const resize180 = () => sharp(svgBuffer).resize(180, 180);

  // 32x32 for favicon.ico — flatten onto dark background (no transparency)
  const png32 = await resize32()
    .flatten({ background: DARK_BG })
    .png()
    .toBuffer();
  const ico = await toIco([png32]);
  fs.writeFileSync(path.join(publicDir, 'favicon.ico'), ico);
  console.log('Created public/favicon.ico');

  // 180x180 for apple-touch-icon — flatten onto dark background
  await resize180()
    .flatten({ background: DARK_BG })
    .png()
    .toFile(path.join(publicDir, 'apple-touch-icon.png'));
  console.log('Created public/apple-touch-icon.png');

  // 32x32 PNG fallback — flatten onto dark background
  await resize32()
    .flatten({ background: DARK_BG })
    .png()
    .toFile(path.join(publicDir, 'favicon-32x32.png'));
  console.log('Created public/favicon-32x32.png');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
