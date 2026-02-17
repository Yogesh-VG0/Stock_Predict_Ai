"""
Module for analyzing short interest data from Nasdaq API.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv
from ..utils.mongodb import MongoDBClient
import aiohttp
import json

# Standard imports (playwright and bs4 methods are deprecated)
from bs4 import BeautifulSoup
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
                                # Look for data in the new Nasdaq API structure
                                table_data = None
                                
                                # First try the new structure: data.shortInterestTable.rows
                                if 'data' in data and isinstance(data['data'], dict):
                                    data_section = data['data']
                                    if 'shortInterestTable' in data_section and isinstance(data_section['shortInterestTable'], dict):
                                        short_table = data_section['shortInterestTable']
                                        if 'rows' in short_table and isinstance(short_table['rows'], list):
                                            table_data = short_table['rows']
                                            logger.info(f"📈 Found data under 'data.shortInterestTable.rows': {len(table_data)} records")
                                
                                # Fallback to old structure checks
                                if not table_data:
                                    possible_data_keys = ['data', 'shortInterestData', 'rows', 'table', 'results']
                                    
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
                                                    # Extract fields from dict record (handle both old and new field names)
                                                    settlement_date = record.get('settlementDate', record.get('date', ''))
                                                    
                                                    # Handle both 'interest' (new) and 'shortInterest' (old) field names
                                                    short_interest = record.get('interest', record.get('shortInterest', record.get('short_interest', 0)))
                                                    
                                                    # Handle both 'avgDailyShareVolume' (new) and 'avgDailyShareVolumeInThousands' (old)
                                                    avg_volume = record.get('avgDailyShareVolume', record.get('avgDailyShareVolumeInThousands', record.get('avg_volume', 0)))
                                                    
                                                    # Handle both 'daysToCover' (new) and 'daysToCoVerShortInterest' (old)
                                                    days_to_cover = record.get('daysToCover', record.get('daysToCoVerShortInterest', record.get('days_to_cover', 0)))
                                                    
                                                    # Clean and convert data
                                                    if isinstance(short_interest, str):
                                                        short_interest = int(short_interest.replace(',', '').replace('$', ''))
                                                    if isinstance(avg_volume, str):
                                                        avg_volume = int(avg_volume.replace(',', '').replace('$', ''))
                                                    if isinstance(days_to_cover, (str, int, float)):
                                                        if isinstance(days_to_cover, str):
                                                            days_to_cover = float(days_to_cover.replace(',', '').replace('$', ''))
                                                        else:
                                                            days_to_cover = float(days_to_cover)
                                                    
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

    def _store_short_interest_raw(self, ticker: str, records: List[Dict]) -> None:
        """Persist raw short interest records in MongoDB for later use."""
        if not self.mongo_client or not records:
            return
        try:
            col = self.mongo_client.db["short_interest_data"]
            for rec in records:
                col.update_one(
                    {"ticker": ticker.upper(), "settlementDate": rec.get("settlementDate", "")},
                    {"$set": {**rec, "ticker": ticker.upper(), "fetched_at": datetime.utcnow()}},
                    upsert=True,
                )
            logger.info(f"Stored {len(records)} short interest records for {ticker}")
        except Exception as e:
            logger.warning(f"Non-critical: failed to store short interest raw data for {ticker}: {e}")

    async def fetch_short_interest(self, ticker: str) -> List[Dict]:
        """
        Fetch short interest data using the new Nasdaq API approach with NYSE fallback.
        This is the main method that should be used.
        """
        try:
            # Check if this is an NYSE stock that needs Finviz backup
            nyse_stocks = [
                'BRK-B', 'LLY', 'WMT', 'JPM', 'V', 'MA', 'XOM', 'ORCL', 'PG', 'JNJ', 
                'UNH', 'HD', 'ABBV', 'KO', 'CRM', 'BAC', 'CVX', 'DIS', 'MRK', 'ADBE',
                'NFLX', 'TMO', 'ACN', 'COST', 'VZ', 'DHR', 'TXN', 'NEE', 'LIN', 'HON',
                'UPS', 'QCOM', 'PM', 'LOW', 'SPGI', 'UNP', 'T', 'RTX', 'IBM', 'INTU',
                'CAT', 'GS', 'DE', 'BKNG', 'AXP', 'ELV', 'LMT', 'SYK', 'GILD', 'MMM',
                'MDLZ', 'CI', 'NOW', 'ISRG', 'TJX', 'CB', 'BLK', 'AMT', 'VRTX', 'ZTS',
                'PLD', 'SCHW', 'MO', 'BSX', 'ADP', 'SHW', 'DUK', 'SO', 'CCI', 'ITW',
                'FI', 'WM', 'MMC', 'AON', 'GD', 'ICE', 'EQIX', 'PNC', 'CL', 'APH',
                'CSX', 'MCK', 'USB', 'TFC', 'NSC', 'EMR', 'COF', 'HUM', 'D', 'PSA',
                'KMB', 'NOC', 'ECL', 'GE', 'WELL', 'SLB', 'EOG', 'TRV', 'HCA', 'AIG'
            ]
            
            if ticker.upper() in nyse_stocks:
                logger.info(f"📊 {ticker} is NYSE-listed, using Finviz for short interest data")
                data = await self.fetch_finviz_short_interest(ticker)
                self._store_short_interest_raw(ticker, data)
                return data
            
            # Use the direct Nasdaq API method for NASDAQ stocks
            logger.info(f"Fetching short interest for {ticker} using Nasdaq API")
            nasdaq_data = await self.fetch_short_interest_direct_api(ticker)
            
            if nasdaq_data:
                logger.info(f"✅ Successfully got {len(nasdaq_data)} records from Nasdaq API")
                self._store_short_interest_raw(ticker, nasdaq_data)
                return nasdaq_data
            else:
                logger.warning(f"❌ No data from Nasdaq API for {ticker}, trying Finviz fallback")
                data = await self.fetch_finviz_short_interest(ticker)
                self._store_short_interest_raw(ticker, data)
                return data
                
        except Exception as e:
            logger.error(f"Error fetching short interest for {ticker}: {e}")
            # Try Finviz as final fallback
            try:
                logger.info(f"🔄 Attempting Finviz fallback for {ticker}")
                return await self.fetch_finviz_short_interest(ticker)
            except Exception as fallback_error:
                logger.error(f"Finviz fallback also failed for {ticker}: {fallback_error}")
                return []

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
        Fetch enhanced short interest data from Finviz with historical data.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Try the short interest specific page first
            url = f"https://finviz.com/quote.ashx?t={ticker}&p=d&ty=si"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        short_data = []
                        logger.info(f"📊 Parsing Finviz short interest data for {ticker}")
                        
                        # First, try to parse the Short Interest History table
                        table = soup.find('table', class_='financials-table')
                        if table:
                            logger.info("✅ Found Short Interest History table")
                            
                            tbody = table.find('tbody')
                            if tbody:
                                rows = tbody.find_all('tr')
                                logger.info(f"📈 Found {len(rows)} historical records")
                                
                                for i, row in enumerate(rows[:10]):  # Limit to 10 most recent
                                    try:
                                        cells = row.find_all('td')
                                        if len(cells) >= 6:
                                            # Extract data from table columns
                                            settlement_date = cells[0].get_text(strip=True)
                                            short_interest_str = cells[1].get_text(strip=True)
                                            shares_float_str = cells[2].get_text(strip=True)
                                            avg_volume_str = cells[3].get_text(strip=True)
                                            short_float_str = cells[4].get_text(strip=True)
                                            short_ratio_str = cells[5].get_text(strip=True)
                                            
                                            # Clean and convert data
                                            short_interest = self._parse_finviz_number(short_interest_str)
                                            shares_float = self._parse_finviz_number(shares_float_str)
                                            avg_volume = self._parse_finviz_number(avg_volume_str)
                                            short_float = float(short_float_str.replace('%', '')) if '%' in short_float_str else 0.0
                                            short_ratio = float(short_ratio_str) if short_ratio_str.replace('.', '').isdigit() else 0.0
                                            
                                            data_point = {
                                                'settlementDate': settlement_date,
                                                'shortInterest': int(short_interest),
                                                'sharesFloat': int(shares_float),
                                                'avgDailyShareVolumeInThousands': int(avg_volume / 1000) if avg_volume > 1000 else int(avg_volume),
                                                'daysToCoVerShortInterest': short_ratio,
                                                'shortFloatPercentage': short_float,
                                                'source': 'finviz_history',
                                                'ticker': ticker,
                                                'recordIndex': i
                                            }
                                            
                                            short_data.append(data_point)
                                            logger.info(f"✅ Parsed historical record {i+1}: {settlement_date} - {short_interest:,.0f} shares ({short_float:.2f}%)")
                                            
                                    except Exception as e:
                                        logger.debug(f"Error parsing historical row {i}: {e}")
                                        continue
                        
                        # If no historical table found, try the main snapshot table
                        if not short_data:
                            logger.info("🔄 No historical table found, trying snapshot data")
                            
                            # Look for snapshot data in the main table
                            tables = soup.find_all('table', class_='snapshot-table2')
                            found_data = {}
                            
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
                                                logger.info(f"📊 Found Short Float: {short_float}%")
                                            except ValueError:
                                                continue
                                        
                                        elif 'Short Interest' in label:
                                            try:
                                                short_interest = self._parse_finviz_number(value)
                                                found_data['short_interest'] = short_interest
                                                logger.info(f"📊 Found Short Interest: {short_interest:,.0f} shares")
                                            except ValueError:
                                                continue
                                        
                                        elif 'Short Ratio' in label or 'Short Interest Ratio' in label:
                                            try:
                                                short_ratio = float(value)
                                                found_data['short_ratio'] = short_ratio
                                                logger.info(f"📊 Found Short Ratio: {short_ratio}")
                                            except ValueError:
                                                continue
                                        
                                        elif 'Shares Outstanding' in label:
                                            try:
                                                shares_outstanding = self._parse_finviz_number(value)
                                                found_data['shares_outstanding'] = shares_outstanding
                                                logger.info(f"📊 Found Shares Outstanding: {shares_outstanding:,.0f}")
                                            except ValueError:
                                                continue
                                        
                                        elif 'Shs Float' in label:
                                            try:
                                                shares_float = self._parse_finviz_number(value)
                                                found_data['shares_float'] = shares_float
                                                logger.info(f"📊 Found Shares Float: {shares_float:,.0f}")
                                            except ValueError:
                                                continue
                            
                            # Create data point if we found anything
                            if found_data:
                                # Calculate short interest if we have short float and shares outstanding/float
                                if 'short_float' in found_data and 'short_interest' not in found_data:
                                    shares_for_calc = found_data.get('shares_float', found_data.get('shares_outstanding', 0))
                                    if shares_for_calc > 0:
                                        short_interest = (found_data['short_float'] / 100) * shares_for_calc
                                        found_data['short_interest'] = short_interest
                                        logger.info(f"📊 Calculated Short Interest: {short_interest:,.0f} shares")
                                
                                short_data.append({
                                    'settlementDate': datetime.utcnow().strftime('%Y-%m-%d'),
                                    'shortInterest': int(found_data.get('short_interest', 0)),
                                    'sharesFloat': int(found_data.get('shares_float', 0)),
                                    'avgDailyShareVolumeInThousands': 0,  # Not available from snapshot
                                    'daysToCoVerShortInterest': found_data.get('short_ratio', 0),
                                    'shortFloatPercentage': found_data.get('short_float', 0),
                                    'sharesOutstanding': int(found_data.get('shares_outstanding', 0)),
                                    'source': 'finviz_snapshot',
                                    'ticker': ticker
                                })
                                
                                logger.info(f"✅ Successfully extracted snapshot data for {ticker} from Finviz")
                            else:
                                logger.warning(f"❌ No short interest data found on Finviz for {ticker}")
                        else:
                            logger.info(f"🎉 Successfully extracted {len(short_data)} historical records for {ticker}")
                        
                        return short_data
                    else:
                        logger.warning(f"❌ Finviz returned status {response.status} for {ticker}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Error fetching short interest from Finviz for {ticker}: {e}")
            return []

    def _parse_finviz_number(self, value_str: str) -> float:
        """Parse number from Finviz with M/B/K suffixes."""
        try:
            value_str = value_str.replace(',', '').strip()
            
            if 'B' in value_str:
                return float(value_str.replace('B', '')) * 1_000_000_000
            elif 'M' in value_str:
                return float(value_str.replace('M', '')) * 1_000_000
            elif 'K' in value_str:
                return float(value_str.replace('K', '')) * 1_000
            else:
                return float(value_str)
        except (ValueError, AttributeError):
            return 0.0

    def get_short_interest_data(self, ticker: str, date) -> Optional[Dict]:
        """
        Get short interest data for a specific date (synchronous method for feature engineering).
        Returns None to indicate short interest is handled in sentiment analysis, not features.
        """
        logger.warning(f"Short interest data requested for features - this is handled in sentiment analysis to avoid duplication")
        return None

    async def analyze_short_interest_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze short interest sentiment using Nasdaq API data.
        """
        try:
            logger.info(f"📊 Starting short interest sentiment analysis for {ticker}")
            
            # Use the direct Nasdaq API method
            short_data = await self.fetch_short_interest(ticker)
            
            if short_data:
                logger.info(f"✅ Successfully got {len(short_data)} records from Nasdaq API for {ticker}")
                latest_data = short_data[0]  # Data is sorted newest first
                short_float = latest_data.get('shortFloatPercentage', 0)
                
                # If we don't have short float percentage from Nasdaq, calculate it
                if short_float == 0 and latest_data.get('shortInterest', 0) > 0:
                    # Estimation using typical shares outstanding for major stocks
                    # For AAPL: ~15.8B shares, for TSLA: ~3.2B, etc.
                    typical_shares = {
                        'AAPL': 15800000000,
                        'TSLA': 3200000000, 
                        'MSFT': 7400000000,
                        'GOOGL': 12600000000,
                        'AMZN': 10800000000
                    }
                    estimated_shares = typical_shares.get(ticker.upper(), 5000000000)  # Default 5B
                    short_float = (latest_data['shortInterest'] / estimated_shares) * 100
                    logger.info(f"📈 Calculated short float from Nasdaq data: {short_float:.2f}%")
                
                data_source = latest_data.get('source', 'nasdaq_api')
                confidence = 0.9  # High confidence in Nasdaq API data
            else:
                logger.warning(f"❌ No short interest data found for {ticker} from Nasdaq API")
                return {
                    'short_interest_sentiment': 0.0,
                    'short_interest_volume': 0,
                    'short_interest_confidence': 0.0,
                    'short_interest_error': 'No data available from Nasdaq API'
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