"""
Module for analyzing short interest data by scraping from Nasdaq.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ..utils.mongodb import MongoDBClient
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import numpy as np
from pathlib import Path
import pickle
import json
import selenium.webdriver as webdriver
import asyncio
from playwright.async_api import async_playwright
import pandas as pd

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BrowserFingerprint:
    """Manages browser fingerprinting to avoid detection."""
    
    def __init__(self):
        """Initialize browser fingerprint with random values."""
        self.ua = UserAgent()
        self._initialize_fingerprint()
    
    def _initialize_fingerprint(self):
        """Initialize fingerprint with random values."""
        self.current_fingerprint = {
            'screen': {
                'width': random.randint(1024, 1920),
                'height': random.randint(768, 1080),
                'colorDepth': random.choice([24, 32]),
                'pixelDepth': random.choice([24, 32])
            },
            'navigator': {
                'language': 'en-US',
                'platform': 'Win32',
                'userAgent': self.ua.random,
                'hardwareConcurrency': random.choice([2, 4, 8]),
                'deviceMemory': random.choice([4, 8, 16])
            },
            'webgl': {
                'vendor': 'Google Inc. (Intel)',
                'renderer': 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0)'
            }
        }
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return self.ua.random
    
    def get_random_viewport(self) -> tuple:
        """Get random viewport dimensions."""
        return (
            self.current_fingerprint['screen']['width'],
            self.current_fingerprint['screen']['height']
        )
    
    def apply_webgl_fingerprint(self, driver):
        """Apply WebGL fingerprint to the browser."""
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': f"""
                    const getParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                        if (parameter === 37445) {{
                            return '{self.current_fingerprint["webgl"]["vendor"]}';
                        }}
                        if (parameter === 37446) {{
                            return '{self.current_fingerprint["webgl"]["renderer"]}';
                        }}
                        return getParameter.apply(this, arguments);
                    }};
                """
            })
        except Exception as e:
            logger.error(f"Failed to apply WebGL fingerprint: {str(e)}")
    
    def apply_navigator_properties(self, driver):
        """Apply navigator properties to the browser."""
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': f"""
                    Object.defineProperty(navigator, 'hardwareConcurrency', {{
                        get: () => {self.current_fingerprint['navigator']['hardwareConcurrency']}
                    }});
                    Object.defineProperty(navigator, 'deviceMemory', {{
                        get: () => {self.current_fingerprint['navigator']['deviceMemory']}
                    }});
                    Object.defineProperty(navigator, 'platform', {{
                        get: () => '{self.current_fingerprint['navigator']['platform']}'
                    }});
                    Object.defineProperty(navigator, 'language', {{
                        get: () => '{self.current_fingerprint['navigator']['language']}'
                    }});
                """
            })
        except Exception as e:
            logger.error(f"Failed to apply navigator properties: {str(e)}")

class HumanBehavior:
    """Simulates human-like behavior"""
    def __init__(self):
        self.last_action_time = datetime.now()
        self.action_history = []
        
    def random_scroll(self, driver):
        """Perform random scrolling"""
        try:
            # Get page height
            page_height = driver.execute_script("return document.body.scrollHeight")
            
            # Generate random scroll points
            scroll_points = np.random.normal(page_height/2, page_height/4, 3)
            scroll_points = np.clip(scroll_points, 0, page_height)
            
            for point in scroll_points:
                driver.execute_script(f"window.scrollTo(0, {point});")
                time.sleep(random.uniform(0.5, 2))
                
            self.action_history.append(('scroll', datetime.now()))
        except:
            pass
            
    def random_mouse_movement(self, driver):
        """Perform random mouse movements"""
        try:
            action = ActionChains(driver)
            
            # Generate random points
            points = [(random.randint(0, 1000), random.randint(0, 700)) for _ in range(5)]
            
            # Move to each point with random delays
            for x, y in points:
                action.move_by_offset(x, y)
                action.pause(random.uniform(0.1, 0.3))
                
            action.perform()
            self.action_history.append(('mouse_move', datetime.now()))
        except:
            pass
            
    def random_wait(self):
        """Wait for a random time between actions"""
        wait_time = random.uniform(1, 5)
        time.sleep(wait_time)
        self.last_action_time = datetime.now()

class SessionManager:
    """Manages browser sessions and cookies"""
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.current_session = None
        
    def save_session(self, driver, session_name: str):
        """Save current session state"""
        session_path = self.session_dir / f"{session_name}.pkl"
        session_data = {
            'cookies': driver.get_cookies(),
            'local_storage': driver.execute_script("return Object.assign({}, window.localStorage);"),
            'session_storage': driver.execute_script("return Object.assign({}, window.sessionStorage);"),
            'timestamp': datetime.now().isoformat()
        }
        with open(session_path, 'wb') as f:
            pickle.dump(session_data, f)
            
    def load_session(self, driver, session_name: str) -> bool:
        """Load saved session state"""
        session_path = self.session_dir / f"{session_name}.pkl"
        if not session_path.exists():
            return False
            
        try:
            with open(session_path, 'rb') as f:
                session_data = pickle.load(f)
                
            # Load cookies
            for cookie in session_data['cookies']:
                try:
                    driver.add_cookie(cookie)
                except:
                    pass
                    
            # Load storage
            driver.execute_script(f"Object.assign(window.localStorage, {json.dumps(session_data['local_storage'])});")
            driver.execute_script(f"Object.assign(window.sessionStorage, {json.dumps(session_data['session_storage'])});")
            
            return True
        except:
            return False

class ShortInterestAnalyzer:
    def __init__(self, mongo_client: MongoDBClient = None, proxy: str = None):
        """Initialize Short Interest Analyzer with enhanced anti-detection."""
        self.mongo_client = mongo_client
        self.proxy = proxy
        self.base_url = "https://www.nasdaq.com/market-activity/stocks/{}/short-interest"
        self.historical_url = "https://www.nasdaq.com/market-activity/stocks/{}/short-interest/historical"
        self.ua = UserAgent()
        self.browser_fingerprint = BrowserFingerprint()
        self.human_behavior = HumanBehavior()
        self.session_manager = SessionManager()
        
    def _create_driver(self) -> webdriver.Chrome:
        """Create and configure Chrome driver with anti-detection measures."""
        options = uc.ChromeOptions()
        
        # Add anti-detection options
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        options.add_argument('--disable-site-isolation-trials')
        
        # Add random user agent
        user_agent = self.browser_fingerprint.get_random_user_agent()
        options.add_argument(f'--user-agent={user_agent}')
        
        # Add random viewport size
        width, height = self.browser_fingerprint.get_random_viewport()
        options.add_argument(f'--window-size={width},{height}')
        
        # Add WebGL fingerprint
        options.add_argument('--use-gl=desktop')
        options.add_argument('--enable-webgl')
        options.add_argument('--enable-webgl2')
        
        # Add language and timezone
        options.add_argument('--lang=en-US')
        options.add_argument('--timezone=America/New_York')
        
        # Add proxy if configured
        if self.proxy:
            options.add_argument(f'--proxy-server={self.proxy}')
        
        try:
            # Create driver with options
            driver = uc.Chrome(options=options)
            
            # Set window size
            driver.set_window_size(width, height)
            
            # Apply WebGL fingerprint
            self.browser_fingerprint.apply_webgl_fingerprint(driver)
            
            # Apply navigator properties
            self.browser_fingerprint.apply_navigator_properties(driver)
            
            return driver
        except Exception as e:
            logger.error(f"Failed to create Chrome driver: {str(e)}")
            raise

    def _handle_cookie_consent(self, driver):
        """Try to accept cookie consent popups if present."""
        consent_selectors = [
            'button#onetrust-accept-btn-handler',
            'button[aria-label="Accept All"]',
            'button[title="Accept"]',
            'button[mode="primary"]',
            'button:contains("Accept All")',
            'button:contains("I Accept")',
            'button:contains("Agree")',
            'button:contains("Continue")',
            'button[aria-label*="Accept"]',
            'button[aria-label*="agree"]',
            'button[title*="Accept"]',
            'button[title*="agree"]',
        ]
        for selector in consent_selectors:
            try:
                # Try both CSS and XPath for robustness
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for btn in elements:
                    if btn.is_displayed() and btn.is_enabled():
                        btn.click()
                        time.sleep(1)
                        return True
            except Exception:
                continue
        # Try a generic accept by text if above fails
        try:
            buttons = driver.find_elements(By.TAG_NAME, 'button')
            for btn in buttons:
                text = btn.text.strip().lower()
                if any(x in text for x in ['accept', 'agree', 'continue']):
                    if btn.is_displayed() and btn.is_enabled():
                        btn.click()
                        time.sleep(1)
                        return True
        except Exception:
            pass
        return False

    def _wait_for_page_load(self, driver, wait):
        """Wait for the page to be fully loaded and dynamic content to be ready."""
        try:
            # Wait for document ready state
            wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
            logger.info("Document ready state complete")
            
            # Wait for jQuery/AJAX to finish (if present)
            try:
                wait.until(lambda d: d.execute_script('return jQuery.active == 0'))
                logger.info("jQuery/AJAX requests completed")
            except:
                pass
            
            # Wait for network to be idle
            try:
                wait.until(lambda d: d.execute_script('return performance.getEntriesByType("resource").every(r => r.duration > 0)'))
                logger.info("Network requests completed")
            except:
                pass
            
            # Wait for any loading indicators to disappear
            try:
                wait.until_not(lambda d: len(d.find_elements(By.CSS_SELECTOR, '.loading, .spinner, [class*="loading"], [class*="spinner"]')) > 0)
                logger.info("Loading indicators disappeared")
            except:
                pass
            
            # Additional wait for dynamic content
            time.sleep(3)
            
        except Exception as e:
            logger.warning(f"Error during page load wait: {e}")
            
    def _fetch_nasdaq_page(self, ticker: str) -> str:
        """
        Fetch short interest page from Nasdaq using undetected Chrome.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            HTML content of the page
        """
        driver = None
        max_retries = 1
        current_retry = 0
        
        while current_retry < max_retries:
        try:
                driver = self._create_driver()
                if not driver:
                    raise Exception("Failed to create Chrome driver")
                
            url = self.base_url.format(ticker.lower())
                logger.info(f"Fetching URL: {url}")
                
                # Try to load saved session
                session_name = f"nasdaq_session_{ticker.lower()}"
                if self.session_manager.load_session(driver, session_name):
                    logger.info(f"Loaded saved session for {ticker}")
                
                # Load page
                driver.get(url)
                logger.info("Page loaded, waiting for content...")
                
                # Wait for page to be ready
                wait = WebDriverWait(driver, 30)
                
                # Wait for initial page load
                self._wait_for_page_load(driver, wait)
                
                # Handle cookie consent if present
                self._handle_cookie_consent(driver)
                time.sleep(2)
                
                # Save initial page source for debugging
                with open('debug_nasdaq_page_initial.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                logger.info("Saved initial page source for debugging")
                
                # Check if we're blocked or need authentication
                if "Access Denied" in driver.page_source or "Please verify you are a human" in driver.page_source:
                    logger.error("Access denied or human verification required")
                    raise Exception("Access denied or human verification required")
                
                # Wait for the table to be present using JavaScript
                try:
                    # Wait for the actual table to load with data
                    wait.until(lambda d: d.execute_script("""
                        const table = document.querySelector('div[class*="short-interest"] table');
                        if (!table) return false;
                        
                        // Check if we have actual data (not just skeleton)
                        const tbody = table.querySelector('tbody');
                        if (!tbody || !tbody.children.length) return false;
                        
                        // Check if first row has actual data
                        const firstRow = tbody.children[0];
                        const cells = firstRow.querySelectorAll('td');
                        if (cells.length < 4) return false;
                        
                        // Check if cells have actual content
                        return Array.from(cells).every(cell => cell.textContent.trim() !== '');
                    """))
                    logger.info("Table found and loaded with data")
                    
                    # Log the actual table structure for debugging
                    table_structure = driver.execute_script("""
                        const table = document.querySelector('div[class*="short-interest"] table');
                        if (!table) return 'No table found';
                        
                        const tbody = table.querySelector('tbody');
                        if (!tbody) return 'No tbody found';
                        
                        const firstRow = tbody.children[0];
                        const cells = firstRow.querySelectorAll('td');
                        
                        return {
                            hasTbody: true,
                            rowCount: tbody.children.length,
                            firstRow: {
                                cellCount: cells.length,
                                cellContents: Array.from(cells).map(cell => cell.textContent.trim()),
                                html: firstRow.innerHTML
                            },
                            hasSkeleton: !!table.querySelector('.jupiter22-c-skeleton')
                        };
                    """)
                    logger.info(f"Table structure: {table_structure}")
                    
                except TimeoutException:
                    logger.error("Table not found or not loaded with data")
                    raise TimeoutException("Table did not load in time")
                
                # Try to scroll to trigger content load
                try:
                    # First try to find the table
                    table = driver.execute_script("""
                        return document.querySelector('div[class*="short-interest"] table');
                    """)
                    
                    if table:
                        # Scroll the table into view
                        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", table)
                        time.sleep(2)
                        
                        # Try to trigger any lazy loading
                        driver.execute_script("""
                            const event = new Event('scroll');
                            window.dispatchEvent(event);
                        """)
                        time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Error during scrolling: {str(e)}")
                
                # Additional wait for dynamic content
                time.sleep(5)
                
                # Get the final page source
                page_source = driver.page_source
                if not page_source:
                    raise Exception("Empty page source received")
                
                # Save final page source for debugging
                with open('debug_nasdaq_page_final.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                logger.info("Saved final page source for debugging")
                
                # Save session for future use
                self.session_manager.save_session(driver, session_name)
                
                return page_source
                
            except Exception as e:
                logger.error(f"Failed to fetch Nasdaq page: {str(e)}")
                if driver:
                    try:
                        # Save error page source for debugging
                        with open('debug_nasdaq_page_error.html', 'w', encoding='utf-8') as f:
                            f.write(driver.page_source)
                        logger.info("Saved error page source for debugging")
                    except:
                        pass
                current_retry += 1
                if current_retry >= max_retries:
            raise
                time.sleep(2)
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                
        raise Exception(f"Failed to fetch page after {max_retries} attempts")
            
    def _parse_short_interest_table(self, html_content: str) -> List[Dict]:
        """
        Parse short interest data from the HTML content.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            List of dictionaries containing short interest data
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the table container
            table_container = soup.find('div', class_='jupiter22-c-short-interest')
            if not table_container:
                logger.error("Could not find table container")
                return []
            
            # Find the actual table
            table = table_container.find('table')
            if not table:
                logger.error("Could not find table")
                return []
                
            # Find all rows (skip header row)
            rows = table.find_all('tr')[1:]  # Skip header row
            if not rows:
                logger.error("Could not find table rows")
                return []
            
            logger.info(f"Found {len(rows)} short interest rows")
            
            # Parse each row
            short_interest_data = []
            for row in rows:
                try:
                    # Get all cells in the row
                    cells = row.find_all('td')
                    if len(cells) < 4:  # We expect 4 columns
                        continue
                    
                    # Get cell text and clean it
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # Skip rows with empty cells
                    if any(not text for text in cell_texts):
                        logger.debug(f"Skipping row with empty cells: {cell_texts}")
                        continue
                    
                    # Parse the data
                    try:
                        settlement_date = cell_texts[0]
                        short_interest = int(cell_texts[1].replace(',', ''))
                        avg_volume = int(cell_texts[2].replace(',', ''))
                        days_to_cover = float(cell_texts[3])
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing cell values: {str(e)}, cells: {cell_texts}")
                        continue
                        
                    # Create data point
                    data_point = {
                        'settlement_date': settlement_date,
                        'short_interest': short_interest,
                        'avg_daily_volume': avg_volume,
                        'days_to_cover': days_to_cover
                    }
                    
                    short_interest_data.append(data_point)
                    logger.debug(f"Parsed row: {data_point}")
                    
                except Exception as e:
                    logger.error(f"Error parsing row: {str(e)}")
                    continue
            
            if not short_interest_data:
                logger.error("No short interest data parsed")
                return []
            
            logger.info(f"Successfully parsed {len(short_interest_data)} short interest records")
            return short_interest_data
            
        except Exception as e:
            logger.error(f"Error parsing short interest table: {str(e)}")
            return []
            
    def _store_short_interest_in_mongodb(self, ticker: str, data: Dict):
        """Store short interest data in MongoDB with validation."""
        try:
            if not self.mongo_client:
                logger.warning("MongoDB client not initialized")
                return False
                
            # Validate data
            required_fields = ['settlement_date', 'short_interest', 'avg_daily_volume']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in short interest data: {data}")
                return False
                
            # Convert settlement_date to datetime if it's a string
            if isinstance(data['settlement_date'], str):
                try:
                    data['settlement_date'] = pd.to_datetime(data['settlement_date'])
                except Exception as e:
                    logger.error(f"Invalid settlement date format: {data['settlement_date']}")
                    return False
                    
            # Check for duplicate data
            existing_data = self.mongo_client.db['short_interest'].find_one({
                'ticker': ticker,
                'settlement_date': data['settlement_date']
            })
            
            if existing_data:
                # Update only if new data is different
                if existing_data.get('short_interest') != data['short_interest'] or \
                   existing_data.get('avg_daily_volume') != data['avg_daily_volume']:
                    self.mongo_client.db['short_interest'].update_one(
                        {'_id': existing_data['_id']},
                        {'$set': data}
                    )
                    logger.info(f"Updated short interest data for {ticker} on {data['settlement_date']}")
            else:
                # Add new data
                data['ticker'] = ticker
                data['fetched_at'] = datetime.utcnow()
                self.mongo_client.db['short_interest'].insert_one(data)
                logger.info(f"Stored new short interest data for {ticker} on {data['settlement_date']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing short interest data in MongoDB: {e}")
            return False
            
    def get_short_interest_data(self, ticker: str, date: datetime) -> Dict:
        """Get short interest data from MongoDB with validation."""
        try:
            if not self.mongo_client:
                logger.warning("MongoDB client not initialized")
                return None
                
            # Convert date to datetime if it's a string
            if isinstance(date, str):
                try:
                    date = pd.to_datetime(date)
                except Exception as e:
                    logger.error(f"Invalid date format: {date}")
                    return None
                    
            # Get the most recent data before or on the given date
            data = self.mongo_client.db['short_interest'].find_one(
                {
                    'ticker': ticker,
                    'settlement_date': {'$lte': date}
                },
                sort=[('settlement_date', -1)]
            )
            
            if data:
                # Remove MongoDB _id field
                data.pop('_id', None)
                return data
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving short interest data from MongoDB: {e}")
            return None

    async def fetch_short_interest(self, ticker: str) -> List[Dict]:
        """
        Fetch short interest data using Playwright.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of dictionaries containing short interest data
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=['--disable-web-security', '--disable-features=IsolateOrigins,site-per-process']
            )
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            try:
                url = self.base_url.format(ticker.lower())
                logger.info(f"Fetching URL: {url}")

                # Navigate to the page with a longer timeout
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                
                # Wait for initial load
                try:
                    await page.wait_for_load_state('domcontentloaded', timeout=30000)
                except Exception as e:
                    logger.warning(f"Initial page load timeout: {e}, continuing anyway")

                # Wait for any loading indicators to disappear
                try:
                    await page.wait_for_selector('.jupiter22-c-skeleton', state='hidden', timeout=10000)
                except Exception as e:
                    logger.warning(f"No skeleton loader found or already disappeared: {e}")

                # Wait for table to be visible and loaded
                try:
                    # First wait for the table container
                    await page.wait_for_selector('div[class*="short-interest"]', state='visible', timeout=30000)
                    logger.info("Found table container")

                    # Scroll to the table to trigger any lazy loading
                    await page.evaluate("""
                        () => {
                            const table = document.querySelector('div[class*="short-interest"]');
                            if (table) {
                                table.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            }
                        }
                    """)
                    
                    # Wait a bit for any lazy loading
                    await page.wait_for_timeout(2000)

                    # Get all rows from the table
                    rows = await page.query_selector_all('div[part="table-row"]')
                    if not rows:
                        raise Exception("No rows found in table")

                    all_data = []
                    for row in rows:
                        cells = await row.query_selector_all('div[part="table-cell"]')
                        if len(cells) >= 4:
                            cell_texts = []
                            for cell in cells[:4]:
                                text = await cell.text_content()
                                text = text.replace('<!--?lit$197389406$-->', '').strip()
                                if text:  # Only add non-empty cells
                                    cell_texts.append(text)
                            
                            if len(cell_texts) == 4:
                                try:
                                    data = {
                                        'settlement_date': cell_texts[0],
                                        'short_interest': int(cell_texts[1].replace(',', '')),
                                        'avg_daily_volume': int(cell_texts[2].replace(',', '')),
                                        'days_to_cover': float(cell_texts[3])
                                    }
                                    all_data.append(data)
                                    logger.info(f"Parsed row: {data}")
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Error parsing row data: {str(e)}, cells: {cell_texts}")
                                    continue

                    if not all_data:
                        raise Exception("No valid data found in table")

                    # Store in MongoDB
                    self._store_short_interest_in_mongodb(ticker, {
                        'data': all_data,
                        'fetched_at': datetime.utcnow()
                    })

                    logger.info(f"Successfully parsed and stored {len(all_data)} rows of data")
                    return all_data
                except Exception as e:
                    logger.error(f"Error waiting for table: {str(e)}")
                    # Save debug information
                    await page.screenshot(path='debug_table_not_found.png')
                    html_content = await page.content()
                    with open('debug_table_not_found.html', 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    raise Exception("Table did not load in time")
            except Exception as e:
                logger.error(f"Error during short interest scraping: {str(e)}")
                raise
            finally:
                await browser.close()

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