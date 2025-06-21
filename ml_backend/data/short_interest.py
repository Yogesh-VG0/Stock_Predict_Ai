"""
Module for analyzing short interest data by scraping from Nasdaq.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ..utils.mongodb import MongoDBClient
from playwright.async_api import async_playwright
import aiohttp
import re

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global lock for coordinating short interest requests
SHORT_INTEREST_LOCK = asyncio.Lock()

class ShortInterestAnalyzer:
    def __init__(self, mongo_client: MongoDBClient = None):
        """Initialize Short Interest Analyzer."""
        self.mongo_client = mongo_client

    async def fetch_short_interest_direct_api(self, ticker: str) -> List[Dict]:
        """
        Fetch short interest data directly from the discovered Nasdaq API endpoint.
        This is much faster and more reliable than web scraping.
        """
        import aiohttp
        import json
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': f'https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/short-interest',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        url = f"https://api.nasdaq.com/api/quote/{ticker.upper()}/short-interest"
        params = {
            'assetClass': 'stocks'
        }
        
        try:
            logger.info(f"🚀 Attempting direct API call to: {url}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"✅ Direct API call successful! Response length: {len(content)}")
                        
                        try:
                            data = json.loads(content)
                            logger.info(f"📊 API Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                            
                            # Parse the API response structure
                            short_data = []
                            
                            if isinstance(data, dict):
                                # Look for data in various possible locations
                                possible_data_keys = ['data', 'shortInterestData', 'rows', 'table', 'results']
                                table_data = None
                                
                                for key in possible_data_keys:
                                    if key in data and data[key]:
                                        table_data = data[key]
                                        logger.info(f"📈 Found data under key '{key}': {type(table_data)}")
                                        break
                                
                                if not table_data and 'data' in data:
                                    # Check if data has nested structure
                                    data_section = data['data']
                                    if isinstance(data_section, dict):
                                        for key in possible_data_keys:
                                            if key in data_section and data_section[key]:
                                                table_data = data_section[key]
                                                logger.info(f"📈 Found nested data under 'data.{key}': {type(table_data)}")
                                                break
                                
                                if table_data:
                                    if isinstance(table_data, list):
                                        # Process array of records
                                        for i, record in enumerate(table_data):
                                            try:
                                                if isinstance(record, dict):
                                                    # Extract fields from dict record
                                                    settlement_date = record.get('settlementDate', record.get('date', ''))
                                                    short_interest = record.get('shortInterest', record.get('short_interest', 0))
                                                    avg_volume = record.get('avgDailyShareVolumeInThousands', record.get('avg_volume', 0))
                                                    days_to_cover = record.get('daysToCoVerShortInterest', record.get('days_to_cover', 0))
                                                    
                                                    # Clean and convert data
                                                    if isinstance(short_interest, str):
                                                        short_interest = int(short_interest.replace(',', '').replace('$', ''))
                                                    if isinstance(avg_volume, str):
                                                        avg_volume = int(avg_volume.replace(',', '').replace('$', ''))
                                                    if isinstance(days_to_cover, str):
                                                        days_to_cover = float(days_to_cover.replace(',', '').replace('$', ''))
                                                    
                                                    data_point = {
                                                        'settlementDate': settlement_date,
                                                        'shortInterest': int(short_interest) if short_interest else 0,
                                                        'avgDailyShareVolumeInThousands': int(avg_volume) if avg_volume else 0,
                                                        'daysToCoVerShortInterest': float(days_to_cover) if days_to_cover else 0.0,
                                                        'source': 'nasdaq_direct_api',
                                                        'ticker': ticker,
                                                        'recordIndex': i,
                                                        'rawRecord': record
                                                    }
                                                    
                                                    short_data.append(data_point)
                                                    logger.info(f"✅ API parsed record {i}: {settlement_date} - {short_interest:,} shares")
                                                    
                                            except Exception as e:
                                                logger.warning(f"⚠️ Error parsing API record {i}: {e}")
                                                continue
                                    
                                    elif isinstance(table_data, dict) and 'rows' in table_data:
                                        # Handle nested rows structure
                                        rows = table_data['rows']
                                        if isinstance(rows, list):
                                            for i, row in enumerate(rows):
                                                try:
                                                    if isinstance(row, dict):
                                                        # Similar processing as above
                                                        settlement_date = row.get('settlementDate', row.get('date', ''))
                                                        short_interest = row.get('shortInterest', 0)
                                                        avg_volume = row.get('avgDailyShareVolumeInThousands', 0)
                                                        days_to_cover = row.get('daysToCoVerShortInterest', 0)
                                                        
                                                        data_point = {
                                                            'settlementDate': settlement_date,
                                                            'shortInterest': int(short_interest) if short_interest else 0,
                                                            'avgDailyShareVolumeInThousands': int(avg_volume) if avg_volume else 0,
                                                            'daysToCoVerShortInterest': float(days_to_cover) if days_to_cover else 0.0,
                                                            'source': 'nasdaq_direct_api_rows',
                                                            'ticker': ticker,
                                                            'recordIndex': i,
                                                            'rawRecord': row
                                                        }
                                                        
                                                        short_data.append(data_point)
                                                        logger.info(f"✅ API parsed nested record {i}: {settlement_date} - {short_interest:,} shares")
                                                        
                                                except Exception as e:
                                                    logger.warning(f"⚠️ Error parsing API nested record {i}: {e}")
                                                    continue
                                
                                # If no structured data found, log the response for debugging
                                if not short_data:
                                    logger.warning(f"📊 No parseable short interest data found in API response")
                                    logger.info(f"🔍 Full API response structure: {json.dumps(data, indent=2)[:1000]}...")
                            
                            if short_data:
                                logger.info(f"🎉 Successfully extracted {len(short_data)} short interest records from direct API for {ticker}")
                                # Sort by date (newest first)
                                short_data.sort(key=lambda x: x['settlementDate'], reverse=True)
                                return short_data[:20]  # Return latest 20 records
                            else:
                                logger.warning(f"❌ No short interest data found in API response for {ticker}")
                                return []
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"❌ Failed to parse API response as JSON: {e}")
                            logger.info(f"📄 Raw response content: {content[:500]}...")
                            return []
                            
                    else:
                        logger.warning(f"❌ API request failed with status {response.status}")
                        if response.status == 403:
                            logger.info("🚫 API access forbidden - may need to use web scraping fallback")
                        return []
                        
        except Exception as e:
            logger.warning(f"❌ Error in direct API call: {e}")
            return []

    async def fetch_short_interest(self, ticker: str) -> List[Dict]:
        """
        Fetch short interest data using multiple strategies:
        1. Direct API call (fastest, most reliable)
        2. Enhanced Playwright scraping (fallback)
        3. Finviz fallback (final fallback)
        """
        
        # Strategy 1: Try direct API first
        logger.info(f"🎯 Strategy 1: Attempting direct Nasdaq API for {ticker}")
        short_data = await self.fetch_short_interest_direct_api(ticker)
        
        if short_data:
            logger.info(f"✅ Direct API successful - returning {len(short_data)} records")
            return short_data
        
        # Strategy 2: Enhanced Playwright scraping fallback
        logger.info(f"🎯 Strategy 2: Falling back to enhanced Playwright scraping for {ticker}")
        short_data = await self.fetch_short_interest_enhanced_playwright(ticker)
        
        if short_data:
            logger.info(f"✅ Playwright scraping successful - returning {len(short_data)} records")
            return short_data
        
        # Strategy 3: Finviz fallback (final fallback)
        logger.info(f"🎯 Strategy 3: Final fallback to Finviz for {ticker}")
        short_data = await self.fetch_finviz_short_interest(ticker)
        
        if short_data:
            logger.info(f"✅ Finviz fallback successful - returning {len(short_data)} records")
            return short_data
        
        logger.warning(f"❌ All strategies failed for {ticker}")
        return []

    async def fetch_short_interest_enhanced_playwright(self, ticker: str) -> List[Dict]:
        """
        Enhanced Playwright scraping method (moved from main fetch_short_interest method).
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # Non-headless for debugging
            
            try:
                page = await browser.new_page()
                
                # Set up network monitoring to catch API calls
                api_responses = []
                
                async def handle_response(response):
                    url = response.url
                    if any(keyword in url.lower() for keyword in ['short', 'interest', 'api', 'data']):
                        try:
                            # Try to get response text
                            if response.status == 200:
                                content = await response.text()
                                api_responses.append({
                                    'url': url,
                                    'status': response.status,
                                    'content': content[:1000],  # First 1000 chars for debugging
                                    'headers': dict(response.headers)
                                })
                                logger.info(f"📡 Captured API response: {url}")
                        except Exception as e:
                            logger.debug(f"Could not capture response from {url}: {e}")
                
                page.on("response", handle_response)
                
                url = f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/short-interest"
                logger.info(f"🔗 Navigating to: {url}")
                logger.info("🖥️ Browser is visible - monitoring dynamic content loading...")
                
                try:
                    await page.goto(url, wait_until="networkidle", timeout=60000)  # Wait for network to be idle
                    logger.info(f"✅ Successfully loaded page for {ticker}")
                    
                    # Wait for initial page load and JavaScript execution
                    await page.wait_for_timeout(5000)
                    
                    # Enhanced scrolling to trigger lazy loading
                    logger.info("📜 Scrolling to trigger dynamic content loading...")
                    
                    # Progressive scrolling to trigger all dynamic content
                    for i in range(10):  # Increased scroll iterations
                        await page.evaluate("window.scrollBy(0, 500)")
                        await page.wait_for_timeout(1500)  # Longer wait between scrolls
                    
                    # Scroll to bottom to ensure all content loads
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(5000)  # Longer wait after bottom scroll
                    
                    # Look for and scroll to short interest section specifically
                    try:
                        # Try multiple selector strategies for the short interest section
                        section_selectors = [
                            'text="Short Interest"',
                            '[data-module="ShortInterest"]',
                            '.short-interest',
                            'div:has-text("Short Interest")',
                            'nsdq-table',
                            '[data-table-id="short-interest"]'
                        ]
                        
                        for selector in section_selectors:
                            try:
                                section = await page.query_selector(selector)
                                if section:
                                    await section.scroll_into_view_if_needed()
                                    logger.info(f"🎯 Found and scrolled to section using: {selector}")
                                    await page.wait_for_timeout(3000)
                                    break
                            except:
                                continue
                    except:
                        logger.info("⚠️ Could not find specific sections, continuing...")
                    
                    # Wait for dynamic content to stabilize with longer timeout
                    await page.wait_for_timeout(8000)
                    
                    # ✅ Try to wait for specific network requests to complete
                    logger.info("🌐 Waiting for potential data API calls...")
                    await page.wait_for_timeout(5000)  # Additional wait for API calls
                    
                    # ✅ Enhanced JavaScript evaluation with more aggressive DOM traversal
                    logger.info("🚀 Using enhanced JavaScript evaluation to extract table data...")
                    
                    # Extract data using multiple JavaScript strategies with longer timeouts
                    row_data = await page.evaluate('''
                        async () => {
                            console.log("🔍 Starting enhanced JavaScript table extraction...");
                            
                            // Wait a bit more for any pending DOM updates
                            await new Promise(resolve => setTimeout(resolve, 2000));
                            
                            // Strategy 1: Look for standard table rows with part attributes
                            let rows = Array.from(document.querySelectorAll('div[part="table-row"]'));
                            console.log(`Strategy 1: Found ${rows.length} rows with part="table-row"`);
                            
                            if (rows.length === 0) {
                                // Strategy 2: Look for any table-like structures
                                rows = Array.from(document.querySelectorAll('div.table-row, tr, [role="row"]'));
                                console.log(`Strategy 2: Found ${rows.length} table-like elements`);
                            }
                            
                            if (rows.length === 0) {
                                // Strategy 3: Look inside any nsdq components
                                const nsdqElements = document.querySelectorAll('nsdq-table, nsdq-my-quotes, [data-module]');
                                console.log(`Strategy 3: Found ${nsdqElements.length} nsdq elements`);
                                
                                for (const elem of nsdqElements) {
                                    const innerRows = elem.querySelectorAll('div, tr, [role="row"]');
                                    rows = rows.concat(Array.from(innerRows));
                                }
                                console.log(`Strategy 3: Total rows after nsdq search: ${rows.length}`);
                            }
                            
                            if (rows.length === 0) {
                                // Strategy 4: Look for any div containing date patterns using more flexible regex
                                const allElements = Array.from(document.querySelectorAll('*'));
                                rows = allElements.filter(elem => {
                                    const text = elem.textContent || '';
                                    return /\\d{1,2}[\/\\-]\\d{1,2}[\/\\-]\\d{4}/.test(text) && 
                                           text.length < 500 && // Avoid large containers
                                           /\\d{1,3}(?:,\\d{3})*/.test(text); // Must also contain formatted numbers
                                });
                                console.log(`Strategy 4: Found ${rows.length} elements with date+number patterns`);
                            }
                            
                            if (rows.length === 0) {
                                // Strategy 5: Last resort - check if there's any structured data in the page
                                const scripts = Array.from(document.querySelectorAll('script'));
                                for (const script of scripts) {
                                    if (script.textContent) {
                                        const text = script.textContent;
                                        if (text.includes('short') && text.includes('interest') && text.includes('{')) {
                                            console.log("Found potential JSON data in script tag");
                                            // Try to extract JSON data
                                            const jsonMatches = text.match(/\\{[^{}]*"[^"]*short[^"]*"[^{}]*\\}/gi);
                                            if (jsonMatches) {
                                                console.log(`Found ${jsonMatches.length} potential JSON objects`);
                                                return jsonMatches.map((match, i) => ({
                                                    cells: [match],
                                                    rowIndex: i,
                                                    strategy: 'json_script',
                                                    rawData: match
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                            
                            const extractedData = [];
                            
                            for (let i = 0; i < Math.min(rows.length, 50); i++) {  // Limit to 50 to avoid processing too much
                                const row = rows[i];
                                
                                // Try to find cells within this row
                                let cells = row.querySelectorAll('div[part="table-cell"], td, th, .cell, [data-cell]');
                                
                                if (cells.length === 0) {
                                    // Fallback: look for any child elements that might be cells
                                    cells = row.querySelectorAll('div, span, td, th');
                                    // Filter out elements that are likely not data cells
                                    cells = Array.from(cells).filter(cell => {
                                        const text = cell.textContent.trim();
                                        return text.length > 0 && text.length < 100 && !/\\n/.test(text);
                                    });
                                }
                                
                                if (cells.length === 0) {
                                    // Last resort: split text content
                                    const text = row.textContent.trim();
                                    if (text && text.length < 200) {
                                        // More sophisticated text splitting
                                        const parts = text.split(/\\s{2,}|\\t|\\|/).filter(p => p.trim());
                                        if (parts.length >= 3) {
                                            cells = parts.map(p => ({ textContent: p.trim() }));
                                        }
                                    }
                                }
                                
                                if (cells.length >= 3) {  // Lowered requirement from 4 to 3
                                    const cellTexts = Array.from(cells).map(cell => 
                                        (cell.textContent || cell.innerText || '').trim()
                                    ).filter(text => text.length > 0);
                                    
                                    // More flexible validation for short interest data
                                    const hasDate = cellTexts.some(text => /\\d{1,2}[\/\\-]\\d{1,2}[\/\\-]\\d{4}/.test(text));
                                    const hasNumbers = cellTexts.filter(text => /\\d{1,3}(?:,\\d{3})*/.test(text)).length >= 2;
                                    
                                    if (hasDate || (hasNumbers && cellTexts.length >= 3)) {
                                        extractedData.push({
                                            cells: cellTexts,
                                            rowIndex: i,
                                            strategy: cells.length > 0 ? 'enhanced_cells' : 'enhanced_text_split',
                                            hasDate: hasDate,
                                            hasNumbers: hasNumbers,
                                            elementTag: row.tagName || 'unknown'
                                        });
                                        console.log(`✅ Enhanced extracted row ${i}: ${cellTexts.slice(0, 4).join(' | ')}`);
                                    }
                                }
                            }
                            
                            console.log(`🎯 Enhanced extraction found: ${extractedData.length} potential rows`);
                            return extractedData;
                        }
                    ''')
                    
                    # ✅ Process extracted data with enhanced validation
                    short_data = []
                    
                    if row_data:
                        logger.info(f"🎯 Enhanced JavaScript extracted {len(row_data)} potential rows")
                        
                        for row_info in row_data:
                            try:
                                cells = row_info['cells']
                                row_index = row_info['rowIndex']
                                strategy = row_info['strategy']
                                
                                # Enhanced data processing
                                if strategy == 'json_script':
                                    # Handle JSON data from script tags
                                    try:
                                        import json
                                        json_data = json.loads(cells[0])
                                        logger.info(f"📊 Found JSON data: {json_data}")
                                        # Process JSON data if it contains relevant information
                                    except:
                                        continue
                                
                                elif len(cells) >= 3:
                                    # Look for date and numeric patterns more flexibly
                                    date_cell = None
                                    numeric_cells = []
                                    
                                    for cell in cells:
                                        if re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}', cell):
                                            date_cell = cell
                                        elif re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', cell):
                                            numeric_cells.append(cell)
                                    
                                    if date_cell and len(numeric_cells) >= 2:
                                        try:
                                            # Clean and convert the data
                                            settlement_date = date_cell
                                            
                                            # Extract numbers from numeric cells
                                            numbers = []
                                            for cell in numeric_cells[:3]:  # Take first 3 numeric values
                                                clean_num = re.sub(r'[^\d.]', '', cell)
                                                if clean_num:
                                                    numbers.append(float(clean_num))
                                            
                                            if len(numbers) >= 2:
                                                short_interest = int(numbers[0])
                                                avg_volume = int(numbers[1]) if len(numbers) > 1 else 0
                                                days_to_cover = numbers[2] if len(numbers) > 2 else 0.0
                                                
                                                data_point = {
                                                    'settlementDate': settlement_date,
                                                    'shortInterest': short_interest,
                                                    'avgDailyShareVolumeInThousands': avg_volume,
                                                    'daysToCoVerShortInterest': days_to_cover,
                                                    'source': f'enhanced_{strategy}',
                                                    'ticker': ticker,
                                                    'rowIndex': row_index,
                                                    'extractionStrategy': strategy,
                                                    'originalCells': cells[:5]  # Store original for debugging
                                                }
                                                
                                                short_data.append(data_point)
                                                logger.info(f"✅ Enhanced parsed row {row_index}: {settlement_date} - {short_interest:,} shares, {days_to_cover:.2f} days to cover")
                                                
                                        except (ValueError, TypeError) as e:
                                            logger.debug(f"⚠️ Error converting enhanced data in row {row_index}: {e}")
                                            continue
                                        
                            except Exception as e:
                                logger.debug(f"❌ Error processing enhanced row: {e}")
                                continue
                    
                    # ✅ Process captured API responses
                    if not short_data and api_responses:
                        logger.info(f"🌐 Processing {len(api_responses)} captured API responses...")
                        for api_resp in api_responses:
                            try:
                                content = api_resp['content']
                                if any(keyword in content.lower() for keyword in ['short', 'interest', 'settlement', 'volume']):
                                    logger.info(f"📊 Found potential short interest data in API: {api_resp['url']}")
                                    # Try to parse JSON or structured data
                                    try:
                                        import json
                                        json_data = json.loads(content)
                                        logger.info(f"📈 API JSON data structure: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
                                        # Process this data based on its structure
                                    except:
                                        # Look for patterns in text content
                                        date_matches = re.findall(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}', content)
                                        if date_matches:
                                            logger.info(f"📅 Found dates in API response: {date_matches[:3]}")
                            except Exception as e:
                                logger.debug(f"Error processing API response: {e}")
                    
                    # ✅ Fallback to BeautifulSoup if all else fails
                    if not short_data:
                        logger.warning("🔄 Enhanced JavaScript extraction failed, falling back to BeautifulSoup parsing...")
                        
                        # Get page content for BeautifulSoup fallback
                        content = await page.content()
                        
                        # Save the page content for debugging
                        with open(f'debug_nasdaq_{ticker}_enhanced_failed.html', 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"💾 Saved page content to debug_nasdaq_{ticker}_enhanced_failed.html")
                        
                        # Use the enhanced BeautifulSoup parser as fallback
                        short_data = self._parse_nasdaq_short_interest_enhanced(content, ticker)
                    
                    # Final validation and sorting
                    if short_data:
                        logger.info(f"🎉 Successfully extracted {len(short_data)} short interest records for {ticker}")
                        # Sort by date (newest first)
                        short_data.sort(key=lambda x: x['settlementDate'], reverse=True)
                    else:
                        logger.warning(f"❌ No short interest data found for {ticker} with any method")
                        logger.info(f"📊 Captured {len(api_responses)} API responses for debugging")
                    
                    return short_data
                    
                except Exception as e:
                    logger.warning(f"❌ Error parsing short interest data for {ticker}: {e}")
                    return []
                    
            finally:
                await browser.close()
                logger.info("🔒 Browser closed")

    def _parse_nasdaq_short_interest(self, html_content: str, ticker: str) -> List[Dict]:
        """
        Parse short interest data from Nasdaq HTML using multiple parsing strategies.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            short_data = []
            
            logger.info("Attempting to parse short interest data...")
            
            # First check if Nasdaq explicitly says data is not available
            error_messages = soup.find_all(text=re.compile(r'short interest.*not available', re.I))
            if error_messages:
                logger.warning("Nasdaq reports: Short interest is currently not available")
                return []
            
            # Strategy 1: Look for the new dynamic table structure with part attributes
            table_rows = soup.find_all('div', {'part': 'table-row', 'class': 'table-row'})
            logger.info(f"Strategy 1 - Found {len(table_rows)} table rows with part='table-row'")
            
            if table_rows:
                for row_idx, row in enumerate(table_rows):
                    try:
                        # Find all table cells in this row with part='table-cell'
                        cells = row.find_all('div', {'part': 'table-cell'})
                        
                        if len(cells) >= 4:
                            # Extract text content from each cell
                            settlement_date = cells[0].get_text(strip=True)
                            short_interest_str = cells[1].get_text(strip=True)
                            avg_volume_str = cells[2].get_text(strip=True)
                            days_to_cover_str = cells[3].get_text(strip=True)
                            
                            # Clean and convert the data
                            try:
                                short_interest = int(short_interest_str.replace(',', ''))
                                avg_volume = int(avg_volume_str.replace(',', ''))
                                days_to_cover = float(days_to_cover_str)
                                
                                data_point = {
                                    'settlementDate': settlement_date,
                                    'shortInterest': short_interest,
                                    'avgDailyShareVolumeInThousands': avg_volume,
                                    'daysToCoVerShortInterest': days_to_cover,
                                    'source': 'nasdaq_dynamic',
                                    'ticker': ticker,
                                    'rowIndex': row_idx
                                }
                                
                                short_data.append(data_point)
                                logger.info(f"Parsed row {row_idx}: {settlement_date} - {short_interest:,} shares, {days_to_cover:.2f} days to cover")
                                
                            except (ValueError, TypeError) as e:
                                logger.debug(f"Error converting data in row {row_idx}: {e}")
                                logger.debug(f"Raw data: date={settlement_date}, short={short_interest_str}, volume={avg_volume_str}, days={days_to_cover_str}")
                                continue
                            
                    except Exception as e:
                        logger.debug(f"Error parsing table row {row_idx}: {e}")
                        continue
            
            # Strategy 2: Look for the older custom div-based table structure (fallback)
            if not short_data:
                logger.info("Strategy 2 - Looking for older table-row structure...")
                old_table_rows = soup.find_all('div', {'part': 'table-row'})
                logger.info(f"Found {len(old_table_rows)} rows with part='table-row' (without class)")
                
                for row in old_table_rows:
                    try:
                        cells = row.find_all('div', {'part': 'table-cell'})
                        
                        if len(cells) >= 4:
                            settlement_date = cells[0].get_text(strip=True)
                            short_interest_str = cells[1].get_text(strip=True)
                            avg_volume_str = cells[2].get_text(strip=True)
                            days_to_cover_str = cells[3].get_text(strip=True)
                            
                            # Clean and convert the data
                            short_interest = int(short_interest_str.replace(',', ''))
                            avg_volume = int(avg_volume_str.replace(',', ''))
                            days_to_cover = float(days_to_cover_str)
                            
                            data_point = {
                                'settlementDate': settlement_date,
                                'shortInterest': short_interest,
                                'avgDailyShareVolumeInThousands': avg_volume,
                                'daysToCoVerShortInterest': days_to_cover,
                                'source': 'nasdaq_fallback',
                                'ticker': ticker
                            }
                            
                            short_data.append(data_point)
                            logger.info(f"Parsed (fallback): {settlement_date} - {short_interest:,} shares")
                            
                    except Exception as e:
                        logger.debug(f"Error parsing fallback table row: {e}")
                        continue
            
            # Strategy 3: Look for nsdq-table elements which might contain the data
            if not short_data:
                logger.info("Strategy 3 - Looking for nsdq-table elements...")
                nsdq_tables = soup.find_all('nsdq-table')
                logger.info(f"Found {len(nsdq_tables)} nsdq-table elements")
                
                for table in nsdq_tables:
                    # Check if this table has data attributes
                    if table.get('data') and table.get('data') != '[object Object]':
                        logger.info("Found nsdq-table with data attribute")
                        # This would require JavaScript execution to get the actual data
                        # For now, we'll skip this approach
            
            # Strategy 4: Look for traditional HTML tables
            if not short_data:
                logger.info("Strategy 4 - Looking for traditional HTML tables...")
                tables = soup.find_all('table')
                logger.info(f"Found {len(tables)} HTML tables")
                
                for table_idx, table in enumerate(tables):
                    logger.info(f"Examining table {table_idx + 1}")
                    
                    # Look for headers to identify the short interest table
                    headers = table.find_all(['th', 'td'])
                    header_text = ' '.join([h.get_text(strip=True).lower() for h in headers[:10]])
                    
                    if any(term in header_text for term in ['settlement', 'short interest', 'days to cover', 'volume']):
                        logger.info(f"Table {table_idx + 1} appears to contain short interest data")
                        
                        rows = table.find_all('tr')[1:]  # Skip header
                        logger.info(f"Found {len(rows)} data rows in table {table_idx + 1}")
                        
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 4:
                                try:
                                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                                    
                                    # Check if this looks like short interest data (has date and numbers)
                                    if any('/' in text or '-' in text for text in cell_texts[:2]):  # Date format
                                        settlement_date = cell_texts[0]
                                        short_interest = int(cell_texts[1].replace(',', ''))
                                        avg_volume = int(cell_texts[2].replace(',', ''))
                                        days_to_cover = float(cell_texts[3])
                                        
                                        data_point = {
                                            'settlementDate': settlement_date,
                                            'shortInterest': short_interest,
                                            'avgDailyShareVolumeInThousands': avg_volume,
                                            'daysToCoVerShortInterest': days_to_cover,
                                            'source': 'nasdaq_table',
                                            'ticker': ticker
                                        }
                                        
                                        short_data.append(data_point)
                                        logger.info(f"Parsed from table: {settlement_date} - {short_interest:,} shares")
                                        
                                except (ValueError, IndexError) as e:
                                    logger.debug(f"Error parsing table row: {e}")
                                    continue
            
            if not short_data:
                logger.warning("No short interest data found with any parsing strategy")
                logger.info("This might indicate that the table is loaded dynamically via JavaScript")
                
                # Log some debugging info
                all_divs_with_part = soup.find_all('div', {'part': True})
                logger.info(f"Found {len(all_divs_with_part)} div elements with 'part' attribute")
                
                table_related_parts = [div.get('part') for div in all_divs_with_part if 'table' in div.get('part', '')]
                if table_related_parts:
                    logger.info(f"Table-related parts found: {set(table_related_parts)}")
            else:
                logger.info(f"Successfully parsed {len(short_data)} short interest records")
                # Sort by date (newest first)
                short_data.sort(key=lambda x: x['settlementDate'], reverse=True)
            
            return short_data
            
        except Exception as e:
            logger.error(f"Error parsing Nasdaq short interest: {e}")
            return []

    def _parse_nasdaq_short_interest_enhanced(self, html_content: str, ticker: str) -> List[Dict]:
        """
        Enhanced BeautifulSoup parser with additional strategies for extracting short interest data.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            short_data = []
            
            logger.info("🔄 Enhanced BeautifulSoup parsing for short interest data...")
            
            # Strategy 1: Look for any element containing date patterns and numeric data
            logger.info("Strategy 1: Searching for elements with date patterns...")
            
            # Find all elements that contain date patterns
            date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')
            potential_rows = []
            
            # Search through all text elements
            for element in soup.find_all(text=date_pattern):
                parent = element.parent
                if parent:
                    # Get all text from this element and siblings
                    text_content = parent.get_text(strip=True, separator=' ')
                    if text_content and ',' in text_content:  # Likely contains formatted numbers
                        potential_rows.append((parent, text_content))
            
            for parent, text_content in potential_rows:
                try:
                    # Split by whitespace and commas to extract components
                    parts = re.split(r'\s+|,', text_content)
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    # Look for a date at the beginning
                    date_found = None
                    numbers = []
                    
                    for i, part in enumerate(parts):
                        if date_pattern.match(part):
                            date_found = part
                            # Extract the next few numeric values
                            for j in range(i+1, min(i+4, len(parts))):
                                clean_num = re.sub(r'[^\d.]', '', parts[j])
                                if clean_num and clean_num.replace('.', '').isdigit():
                                    numbers.append(clean_num)
                            break
                    
                    if date_found and len(numbers) >= 3:
                        try:
                            short_interest = int(float(numbers[0]))
                            avg_volume = int(float(numbers[1]))
                            days_to_cover = float(numbers[2])
                            
                            data_point = {
                                'settlementDate': date_found,
                                'shortInterest': short_interest,
                                'avgDailyShareVolumeInThousands': avg_volume,
                                'daysToCoVerShortInterest': days_to_cover,
                                'source': 'enhanced_text_pattern',
                                'ticker': ticker,
                                'rawText': text_content[:100]  # Store for debugging
                            }
                            
                            short_data.append(data_point)
                            logger.info(f"✅ Extracted from text pattern: {date_found} - {short_interest:,} shares")
                            
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Failed to convert extracted numbers: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error processing potential row: {e}")
                    continue
            
            # Strategy 2: Look for structured data in script tags or data attributes
            if not short_data:
                logger.info("Strategy 2: Searching for JSON data in script tags...")
                
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string:
                        # Look for JSON-like structures
                        text = script.string
                        if 'short' in text.lower() and 'interest' in text.lower():
                            # Try to extract JSON data
                            import json
                            json_matches = re.findall(r'\{[^{}]*"[^"]*short[^"]*"[^{}]*\}', text, re.IGNORECASE)
                            for match in json_matches:
                                try:
                                    data = json.loads(match)
                                    logger.info(f"Found potential JSON data: {data}")
                                    # Process if it contains relevant data
                                except:
                                    continue
            
            # Strategy 3: Look for table-like structures without proper table tags
            if not short_data:
                logger.info("Strategy 3: Searching for table-like div structures...")
                
                # Find divs that might represent table rows
                all_divs = soup.find_all('div')
                potential_table_rows = []
                
                for div in all_divs:
                    text = div.get_text(strip=True)
                    # Check if this div contains data that looks like a table row
                    if (date_pattern.search(text) and 
                        len(re.findall(r'\d{1,3}(?:,\d{3})*', text)) >= 2):  # Has date and multiple numbers
                        potential_table_rows.append(div)
                
                for div in potential_table_rows:
                    try:
                        # Extract all numbers and the date
                        text = div.get_text(strip=True)
                        date_match = date_pattern.search(text)
                        if date_match:
                            date_str = date_match.group()
                            
                            # Find all comma-separated numbers
                            number_matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
                            
                            if len(number_matches) >= 3:
                                try:
                                    # Assume order: short interest, avg volume, days to cover
                                    short_interest = int(number_matches[0].replace(',', ''))
                                    avg_volume = int(number_matches[1].replace(',', ''))
                                    days_to_cover = float(number_matches[2].replace(',', ''))
                                    
                                    data_point = {
                                        'settlementDate': date_str,
                                        'shortInterest': short_interest,
                                        'avgDailyShareVolumeInThousands': avg_volume,
                                        'daysToCoVerShortInterest': days_to_cover,
                                        'source': 'enhanced_div_structure',
                                        'ticker': ticker,
                                        'rawText': text[:100]
                                    }
                                    
                                    short_data.append(data_point)
                                    logger.info(f"✅ Extracted from div structure: {date_str} - {short_interest:,} shares")
                                    
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Failed to convert div data: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.debug(f"Error processing div row: {e}")
                        continue
            
            if short_data:
                logger.info(f"🎉 Enhanced parser found {len(short_data)} short interest records")
                # Sort by date (newest first)
                short_data.sort(key=lambda x: x['settlementDate'], reverse=True)
            else:
                logger.warning("❌ Enhanced parser found no short interest data")
                
                # Debug information
                logger.info("Debug: Looking for any short interest related text...")
                short_text = soup.find_all(text=re.compile(r'short.*interest', re.IGNORECASE))
                if short_text:
                    logger.info(f"Found {len(short_text)} elements containing 'short interest'")
                    for i, text in enumerate(short_text[:3]):  # Show first 3
                        logger.info(f"  {i+1}: {str(text).strip()[:100]}...")
            
            return short_data
            
        except Exception as e:
            logger.error(f"Error in enhanced Nasdaq parser: {e}")
            return []

    def _fallback_parse_nasdaq(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Fallback parser for alternative table structures."""
        try:
            short_data = []
            
            # Look for any table with short interest data
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            cell_texts = [cell.get_text(strip=True) for cell in cells]
                            
                            # Check if this looks like short interest data
                            if any('/' in text for text in cell_texts[:2]):  # Date format
                                settlement_date = cell_texts[0]
                                short_interest = int(cell_texts[1].replace(',', ''))
                                avg_volume = int(cell_texts[2].replace(',', ''))
                                days_to_cover = float(cell_texts[3])
                                
                                data_point = {
                                    'settlementDate': settlement_date,
                                    'shortInterest': short_interest,
                                    'avgDailyShareVolumeInThousands': avg_volume,
                                    'daysToCoVerShortInterest': days_to_cover,
                                    'source': 'nasdaq_fallback',
                                    'ticker': ticker
                                }
                                
                                short_data.append(data_point)
                                
                        except (ValueError, IndexError):
                            continue
            
            return short_data
            
        except Exception as e:
            logger.error(f"Error in fallback parser: {e}")
            return []

    async def fetch_finviz_short_interest(self, ticker: str) -> List[Dict]:
        """
        Enhanced alternative short interest data fetching using Finviz (more reliable).
        """
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            logger.info(f"Fetching Finviz data for {ticker}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        short_data = []
                        found_data = {}
                        
                        logger.info("Parsing Finviz data...")
                        
                        # Strategy 1: Look for the main statistics table
                        tables = soup.find_all('table', class_='snapshot-table2')
                        
                        for table in tables:
                            rows = table.find_all('tr')
                            for row in rows:
                                cells = row.find_all('td')
                                
                                # Process pairs of cells (label, value)
                                for i in range(0, len(cells) - 1, 2):
                                    label = cells[i].get_text(strip=True)
                                    value = cells[i + 1].get_text(strip=True)
                                    
                                    if 'Short Float' in label:
                                        try:
                                            short_float = float(value.replace('%', ''))
                                            found_data['short_float'] = short_float
                                            logger.info(f"Found Short Float: {short_float}%")
                                        except ValueError:
                                            continue
                                    
                                    elif 'Short Interest' in label:
                                        try:
                                            short_interest_text = value.replace(',', '')
                                            if 'M' in short_interest_text:
                                                short_interest = float(short_interest_text.replace('M', '')) * 1000000
                                            elif 'K' in short_interest_text:
                                                short_interest = float(short_interest_text.replace('K', '')) * 1000
                                            else:
                                                short_interest = float(short_interest_text)
                                            found_data['short_interest'] = short_interest
                                            logger.info(f"Found Short Interest: {short_interest:,.0f} shares")
                                        except ValueError:
                                            continue
                                    
                                    elif 'Short Ratio' in label or 'Short Interest Ratio' in label:
                                        try:
                                            short_ratio = float(value)
                                            found_data['short_ratio'] = short_ratio
                                            logger.info(f"Found Short Ratio: {short_ratio}")
                                        except ValueError:
                                            continue
                                    
                                    elif 'Shares Outstanding' in label:
                                        try:
                                            shares_text = value.replace(',', '')
                                            if 'B' in shares_text:
                                                shares_outstanding = float(shares_text.replace('B', '')) * 1000000000
                                            elif 'M' in shares_text:
                                                shares_outstanding = float(shares_text.replace('M', '')) * 1000000
                                            else:
                                                shares_outstanding = float(shares_text)
                                            found_data['shares_outstanding'] = shares_outstanding
                                            logger.info(f"Found Shares Outstanding: {shares_outstanding:,.0f}")
                                        except ValueError:
                                            continue
                        
                        # Strategy 2: If main table didn't work, try alternative selectors
                        if not found_data:
                            logger.info("Trying alternative Finviz parsing...")
                            
                            # Look for any table cells containing short-related data
                            all_cells = soup.find_all('td')
                            for i, cell in enumerate(all_cells):
                                cell_text = cell.get_text(strip=True)
                                
                                if 'Short Float' in cell_text and i + 1 < len(all_cells):
                                    next_cell = all_cells[i + 1]
                                    short_float_text = next_cell.get_text(strip=True)
                                    try:
                                        short_float = float(short_float_text.replace('%', ''))
                                        found_data['short_float'] = short_float
                                        logger.info(f"Found Short Float (alt): {short_float}%")
                                    except ValueError:
                                        continue
                                
                                elif 'Short Interest' in cell_text and i + 1 < len(all_cells):
                                    next_cell = all_cells[i + 1]
                                    short_interest_text = next_cell.get_text(strip=True)
                                    try:
                                        if 'M' in short_interest_text:
                                            short_interest = float(short_interest_text.replace('M', '')) * 1000000
                                        elif 'K' in short_interest_text:
                                            short_interest = float(short_interest_text.replace('K', '')) * 1000
                                        else:
                                            short_interest = float(short_interest_text.replace(',', ''))
                                        found_data['short_interest'] = short_interest
                                        logger.info(f"Found Short Interest (alt): {short_interest:,.0f} shares")
                                    except ValueError:
                                        continue
                        
                        # Create data point if we found anything
                        if found_data:
                            # Calculate short interest if we have short float and shares outstanding
                            if 'short_float' in found_data and 'shares_outstanding' in found_data and 'short_interest' not in found_data:
                                short_interest = (found_data['short_float'] / 100) * found_data['shares_outstanding']
                                found_data['short_interest'] = short_interest
                                logger.info(f"Calculated Short Interest: {short_interest:,.0f} shares")
                            
                            short_data.append({
                                'settlementDate': datetime.utcnow().strftime('%Y-%m-%d'),
                                'shortInterest': int(found_data.get('short_interest', 0)),
                                'avgDailyShareVolumeInThousands': 0,  # Not available from Finviz
                                'daysToCoVerShortInterest': found_data.get('short_ratio', 0),
                                'shortFloatPercentage': found_data.get('short_float', 0),
                                'sharesOutstanding': int(found_data.get('shares_outstanding', 0)),
                                'source': 'finviz',
                                'ticker': ticker
                            })
                            
                            logger.info(f"Successfully extracted enhanced short interest data for {ticker} from Finviz")
                            logger.info(f"Short Float: {found_data.get('short_float', 0):.2f}%")
                            logger.info(f"Short Interest: {found_data.get('short_interest', 0):,.0f} shares")
                            logger.info(f"Short Ratio: {found_data.get('short_ratio', 0):.2f}")
                        else:
                            logger.warning(f"No short interest data found on Finviz for {ticker}")
                        
                        return short_data
                    else:
                        logger.warning(f"Finviz returned status {response.status} for {ticker}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching short interest from Finviz for {ticker}: {e}")
            return []

    async def analyze_short_interest_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze short interest sentiment with improved fallback strategy.
        """
        try:
            logger.info(f"Starting short interest sentiment analysis for {ticker}")
            
            # Try Finviz first since it's more reliable for current data
            logger.info(f"Trying Finviz (primary source) for {ticker} short interest")
            short_data = await self.fetch_finviz_short_interest(ticker)
            
            # If Finviz works, use it
            if short_data:
                logger.info(f"Successfully got data from Finviz for {ticker}")
                latest_data = short_data[-1]
                short_float = latest_data.get('shortFloatPercentage', 0)
                data_source = 'finviz'
                confidence = 0.8  # High confidence in Finviz data
            else:
                # Try Nasdaq as fallback
                logger.info(f"Finviz failed, trying Nasdaq for {ticker} short interest")
                short_data = await self.fetch_short_interest(ticker)
                
                if short_data:
                    logger.info(f"Successfully got data from Nasdaq for {ticker}")
                    latest_data = short_data[-1]
                    short_float = latest_data.get('shortFloatPercentage', 0)
                    
                    # If we don't have short float percentage from Nasdaq, calculate it
                    if short_float == 0 and latest_data.get('shortInterest', 0) > 0:
                        # Rough estimation for AAPL: assume shares outstanding is ~15.8B
                        estimated_shares_outstanding = 15800000000  # 15.8B shares for AAPL
                        short_float = (latest_data['shortInterest'] / estimated_shares_outstanding) * 100
                        logger.info(f"Calculated short float from Nasdaq data: {short_float:.2f}%")
                    
                    data_source = latest_data.get('source', 'nasdaq')
                    confidence = 0.9  # High confidence in Nasdaq data when available
                else:
                    logger.warning(f"No short interest data found for {ticker} from any source")
                    return {
                        'short_interest_sentiment': 0.0,
                        'short_interest_volume': 0,
                        'short_interest_confidence': 0.0,
                        'short_interest_error': 'No data available from any source (Finviz or Nasdaq)'
                    }
            
            # Calculate sentiment from short float percentage
            # Short interest sentiment logic (same as before):
            if short_float >= 20:  # Very high short interest
                sentiment = -0.8
            elif short_float >= 10:  # High short interest
                sentiment = -0.5
            elif short_float >= 5:  # Moderate short interest
                sentiment = -0.2
            elif short_float >= 2:  # Low short interest
                sentiment = 0.2
            else:  # Very low short interest
                sentiment = 0.5
            
            logger.info(f"Short interest sentiment for {ticker}: {sentiment:.3f} (short float: {short_float:.2f}%)")
            
            return {
                'short_interest_sentiment': sentiment,
                'short_interest_volume': len(short_data),
                'short_interest_confidence': confidence,
                'short_float_percentage': short_float,
                'data_source': data_source,
                'short_interest_error': None,
                'short_interest_data': latest_data.get('shortInterest', 0),
                'days_to_cover': latest_data.get('daysToCoVerShortInterest', 0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing short interest sentiment for {ticker}: {e}")
            return {
                'short_interest_sentiment': 0.0,
                'short_interest_volume': 0,
                'short_interest_confidence': 0.0,
                'short_interest_error': str(e)
            }

async def main():
    analyzer = ShortInterestAnalyzer()
    ticker = "AAPL"
    try:
        data = await analyzer.fetch_short_interest(ticker)
        print(f"Successfully fetched short interest data for {ticker}:")
        print(data)
    except Exception as e:
        print(f"Error fetching short interest data: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 