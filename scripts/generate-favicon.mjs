#!/usr/bin/env node
/**
 * Generate favicon.ico and apple-touch-icon.png from app/icon.svg
 * Run: node scripts/generate-favicon.mjs
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import toIco from 'to-ico';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const svgPath = path.join(root, 'app', 'icon.svg');
const publicDir = path.join(root, 'public');

if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });

const svgBuffer = fs.readFileSync(svgPath);

async function main() {
  // 32x32 for favicon.ico
  const png32 = await sharp(svgBuffer)
    .resize(32, 32)
    .png()
    .toBuffer();
  const ico = await toIco([png32]);
  fs.writeFileSync(path.join(publicDir, 'favicon.ico'), ico);
  console.log('Created public/favicon.ico');

  // 180x180 for apple-touch-icon
  await sharp(svgBuffer)
    .resize(180, 180)
    .png()
    .toFile(path.join(publicDir, 'apple-touch-icon.png'));
  console.log('Created public/apple-touch-icon.png');

  // 32x32 PNG as fallback
  await sharp(svgBuffer)
    .resize(32, 32)
    .png()
    .toFile(path.join(publicDir, 'favicon-32x32.png'));
  console.log('Created public/favicon-32x32.png');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
