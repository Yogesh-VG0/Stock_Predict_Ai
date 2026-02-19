"""
Economic Calendar data fetcher for market-moving events.
Scrapes Investing.com and maps events to affected tickers.
Uses FMP data from sentiment pipeline stored in MongoDB.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time
from bs4 import BeautifulSoup
import json
import re
import os
import aiohttp
import asyncio
import traceback
import random
import sys
import platform
import pickle
import hashlib
from pathlib import Path
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# Conditional selenium imports - only import if available (not in CI/prod)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.action_chains import ActionChains
    import undetected_chromedriver as uc
    from fake_useragent import UserAgent
    import cloudscraper
    from selenium_stealth import stealth
    from selenium.webdriver.common.proxy import Proxy, ProxyType
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    # Create dummy classes for type hints
    webdriver = None
    Options = None
    By = None
    WebDriverWait = None
    EC = None
    ChromeDriverManager = None
    Service = None
    ActionChains = None
    uc = None
    UserAgent = None
    cloudscraper = None
    stealth = None
    Proxy = None
    ProxyType = None
    WebDriver = None
    Keys = None
    TimeoutException = Exception
    NoSuchElementException = Exception
    ElementClickInterceptedException = Exception

logger = logging.getLogger(__name__)

# Event impact mapping to tickers
EVENT_TICKER_MAPPING = {
    # Federal Reserve Events (High Impact)
    'fomc': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],  # Financial sector
    'fed interest rate': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc statement': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc press conference': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc meeting minutes': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc economic projections': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fed chair powell': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    
    # Employment Data (High Impact)
    'nonfarm payrolls': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'unemployment rate': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'jobless claims': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'adp nonfarm': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'jolts': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'average hourly earnings': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Inflation Indicators (High Impact)
    'cpi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'core cpi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'ppi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'GE', 'BA', 'CAT', 'DE', 'HON'],
    'core pce': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Retail and Consumer Data (Medium Impact)
    'retail sales': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'core retail sales': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'consumer confidence': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    
    # Manufacturing and Industrial Data (Medium Impact)
    'ism manufacturing': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'ism non-manufacturing': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'durable goods': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'philadelphia fed': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'chicago pmi': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    
    # Housing Data (Medium Impact)
    'new home sales': ['HD', 'LOW', 'LMT', 'RTX', 'BA', 'SPG', 'AMT'],
    'existing home sales': ['HD', 'LOW', 'LMT', 'RTX', 'BA', 'SPG', 'AMT'],
    
    # Energy Data (Medium Impact)
    'crude oil': ['XOM', 'CVX', 'COP'],
    
    # GDP and Economic Growth (High Impact)
    'gdp': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'GOOGL', 'META', 'TSLA', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Tech and Innovation Events
    'semiconductor': ['NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'ai': ['NVDA', 'AMD', 'INTC', 'MSFT', 'GOOG', 'GOOGL', 'META', 'AAPL'],
    'cloud': ['MSFT', 'AMZN', 'GOOG', 'GOOGL', 'ORCL', 'CRM', 'NOW'],
    
    # Healthcare Events
    'healthcare': ['JNJ', 'UNH', 'ABBV', 'ABT', 'MRK', 'PFE', 'BMY', 'MDT', 'ISRG'],
    'pharma': ['JNJ', 'ABBV', 'ABT', 'MRK', 'PFE', 'BMY', 'LLY'],
    
    # Financial Events
    'banking': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF'],
    'insurance': ['UNH', 'AIG', 'MET'],
    
    # Consumer Events
    'consumer': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'discretionary': ['AMZN', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'DIS'],
    
    # Industrial Events
    'industrial': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT'],
    
    # Real Estate Events
    'real estate': ['SPG', 'AMT'],
    
    # Telecom Events
    'telecom': ['T', 'TMUS', 'VZ', 'CMCSA', 'CHTR'],
    
    # Utilities Events
    'utilities': ['DUK', 'SO', 'NEE']
}

# Event importance scoring
EVENT_IMPORTANCE = {
    'high': 1.0,
    'medium': 0.6,
    'low': 0.3
}

class ProxyManager:
    """Manages proxy rotation and validation"""
    def __init__(self):
        self.proxies = []
        self.current_proxy = None
        self.last_rotation = datetime.now()
        self.rotation_interval = timedelta(minutes=5)
        
    def add_proxy(self, proxy: str):
        """Add a proxy to the pool"""
        if self._validate_proxy(proxy):
            self.proxies.append(proxy)
            
    def get_proxy(self) -> Optional[str]:
        """Get next proxy with rotation"""
        if not self.proxies:
            return None
            
        if (datetime.now() - self.last_rotation) > self.rotation_interval:
            self.current_proxy = random.choice(self.proxies)
            self.last_rotation = datetime.now()
            
        return self.current_proxy
        
    def _validate_proxy(self, proxy: str) -> bool:
        """Validate proxy is working"""
        try:
            response = requests.get('https://www.investing.com', 
                                 proxies={'http': proxy, 'https': proxy},
                                 timeout=10)
            return response.status_code == 200
        except:
            return False

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

class BrowserFingerprint:
    """Generates and manages browser fingerprints"""
    def __init__(self):
        self.fingerprints = []
        self.current_fingerprint = None
        
    def generate_fingerprint(self) -> Dict[str, Any]:
        """Generate a realistic browser fingerprint"""
        # Common screen resolutions
        resolutions = [
            (1920, 1080), (1366, 768), (1536, 864),
            (1440, 900), (1280, 720), (1600, 900)
        ]
        
        # Common color depths
        color_depths = [24, 32, 48]
        
        # Common timezones
        timezones = [
            'America/New_York', 'America/Chicago',
            'America/Los_Angeles', 'America/Phoenix'
        ]
        
        # Generate fingerprint
        fingerprint = {
            'screen': {
                'width': random.choice(resolutions)[0],
                'height': random.choice(resolutions)[1],
                'colorDepth': random.choice(color_depths),
                'pixelDepth': random.choice(color_depths)
            },
            'navigator': {
                'userAgent': UserAgent().random,
                'language': random.choice(['en-US', 'en-GB', 'en']),
                'platform': random.choice(['Win32', 'MacIntel', 'Linux x86_64']),
                'hardwareConcurrency': random.choice([2, 4, 8, 16]),
                'deviceMemory': random.choice([4, 8, 16, 32]),
                'timezone': random.choice(timezones)
            },
            'webgl': {
                'vendor': random.choice(['Google Inc.', 'Intel Inc.', 'NVIDIA Corporation']),
                'renderer': random.choice([
                    'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0)',
                    'ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0)',
                    'ANGLE (AMD, AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)'
                ])
            },
            'audio': {
                'sampleRate': random.choice([44100, 48000, 96000]),
                'channelCount': random.choice([2, 4, 6])
            }
        }
        
        self.current_fingerprint = fingerprint
        self.fingerprints.append(fingerprint)
        return fingerprint
        
    def apply_fingerprint(self, driver):
        """Apply fingerprint to browser"""
        if not self.current_fingerprint:
            self.generate_fingerprint()
            
        # Apply screen settings
        driver.execute_script(f"""
            Object.defineProperty(window.screen, 'width', {{value: {self.current_fingerprint['screen']['width']}}});
            Object.defineProperty(window.screen, 'height', {{value: {self.current_fingerprint['screen']['height']}}});
            Object.defineProperty(window.screen, 'colorDepth', {{value: {self.current_fingerprint['screen']['colorDepth']}}});
            Object.defineProperty(window.screen, 'pixelDepth', {{value: {self.current_fingerprint['screen']['pixelDepth']}}});
        """)
        
        # Apply navigator settings
        driver.execute_script(f"""
            Object.defineProperty(navigator, 'hardwareConcurrency', {{value: {self.current_fingerprint['navigator']['hardwareConcurrency']}}});
            Object.defineProperty(navigator, 'deviceMemory', {{value: {self.current_fingerprint['navigator']['deviceMemory']}}});
            Object.defineProperty(navigator, 'platform', {{value: '{self.current_fingerprint['navigator']['platform']}'}});
            Object.defineProperty(navigator, 'language', {{value: '{self.current_fingerprint['navigator']['language']}'}});
        """)

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
            
    def random_typing(self, driver, element, text: str):
        """Simulate human typing"""
        try:
            for char in text:
                element.send_keys(char)
                time.sleep(random.uniform(0.1, 0.3))
            self.action_history.append(('typing', datetime.now()))
        except:
            pass
            
    def random_wait(self):
        """Wait for a random time between actions"""
        wait_time = random.uniform(1, 5)
        time.sleep(wait_time)
        self.last_action_time = datetime.now()

class EconomicCalendar:
    """
    Fetches and processes economic calendar events.
    Maps events to affected tickers for event-driven feature engineering.
    Uses FMP data from sentiment pipeline stored in MongoDB.
    """
    
    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client
        self.rate_limit_delay = 2.0
        self.selenium_available = SELENIUM_AVAILABLE
        
        # Only initialize selenium-dependent components if selenium is available
        if SELENIUM_AVAILABLE:
            try:
                self.proxy_manager = ProxyManager()
                self.session_manager = SessionManager()
                self.fingerprint_manager = BrowserFingerprint()
                self.human_behavior = HumanBehavior()
                
                # Initialize proxy list (you should add your proxies here)
                self.proxy_manager.add_proxy("http://proxy1.example.com:8080")
                self.proxy_manager.add_proxy("http://proxy2.example.com:8080")
                
                # Setup Chrome options for undetected scraping
                self.chrome_options = uc.ChromeOptions()
            except Exception as e:
                logger.warning("Failed to initialize selenium components: %s", e)
                self.selenium_available = False
        else:
            logger.info("Selenium not available - economic calendar scraping disabled (CI/prod mode)")
            self.proxy_manager = None
            self.session_manager = None
            self.fingerprint_manager = None
            self.human_behavior = None
            self.chrome_options = None
        
        # Instance-level cache to prevent multiple fetches per session
        self._events_cache = {
            'data': None,
            'timestamp': None,
            'cache_duration': timedelta(hours=6)
        }
        
        # Only configure Chrome options if selenium is available
        if self.selenium_available and self.chrome_options is not None:
            try:
                # Basic settings
                self.chrome_options.add_argument('--no-sandbox')
                self.chrome_options.add_argument('--disable-dev-shm-usage')
                self.chrome_options.add_argument('--disable-gpu')
                self.chrome_options.add_argument('--window-size=1920,1080')
                self.chrome_options.add_argument('--start-maximized')
                
                # Security and CSP settings
                self.chrome_options.add_argument('--disable-web-security')
                self.chrome_options.add_argument('--allow-running-insecure-content')
                self.chrome_options.add_argument('--disable-site-isolation-trials')
                self.chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
                self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                
                # WebGL settings
                self.chrome_options.add_argument('--enable-unsafe-swiftshader')
                self.chrome_options.add_argument('--ignore-gpu-blocklist')
                self.chrome_options.add_argument('--enable-gpu-rasterization')
                
                # Network settings
                self.chrome_options.add_argument('--dns-prefetch-disable')
                self.chrome_options.add_argument('--disable-background-networking')
                self.chrome_options.add_argument('--disable-default-apps')
                self.chrome_options.add_argument('--disable-extensions')
                self.chrome_options.add_argument('--disable-sync')
                self.chrome_options.add_argument('--disable-translate')
                self.chrome_options.add_argument('--hide-scrollbars')
                self.chrome_options.add_argument('--metrics-recording-only')
                self.chrome_options.add_argument('--mute-audio')
                self.chrome_options.add_argument('--no-first-run')
                self.chrome_options.add_argument('--safebrowsing-disable-auto-update')
                
                # Random user agent
                ua = UserAgent()
                self.chrome_options.add_argument(f'--user-agent={ua.random}')
            except Exception as e:
                logger.warning("Failed to configure Chrome options: %s", e)
                self.selenium_available = False
        
        # Additional preferences
        prefs = {
            'profile.default_content_setting_values': {
                'notifications': 2,
                'images': 2,
                'javascript': 1,
                'cookies': 1,
                'plugins': 2,
                'popups': 2,
                'geolocation': 2,
                'media_stream': 2,
            },
            'profile.managed_default_content_settings': {
                'javascript': 1,
                'cookies': 1,
            },
            'profile.cookie_controls_mode': 0,
            'credentials_enable_service': False,
            'profile.password_manager_enabled': False,
            'profile.default_content_settings.popups': 0,
            'profile.managed_default_content_settings.images': 2,
            'profile.default_content_setting_values.notifications': 2,
            'profile.managed_default_content_settings.javascript': 1,
            'profile.managed_default_content_settings.cookies': 1,
            'profile.managed_default_content_settings.plugins': 2,
            'profile.managed_default_content_settings.popups': 2,
            'profile.managed_default_content_settings.geolocation': 2,
            'profile.managed_default_content_settings.media_stream': 2,
                }
                self.chrome_options.add_experimental_option('prefs', prefs)
                
                # Add random viewport size
                viewport_width = random.randint(1024, 1920)
                viewport_height = random.randint(768, 1080)
                self.chrome_options.add_argument(f'--window-size={viewport_width},{viewport_height}')
            except Exception as e:
                logger.warning("Failed to configure Chrome options: %s", e)
                self.selenium_available = False
        
    def _create_undetected_driver(self):
        """Create an undetected Chrome driver instance"""
        if not self.selenium_available:
            logger.warning("Selenium not available - cannot create Chrome driver")
            return None
        try:
            # Create new ChromeOptions instance each time
            chrome_options = uc.ChromeOptions()
            
            # Basic settings
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
            
            # Security and CSP settings
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--disable-site-isolation-trials')
            chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            
            # WebGL settings
            chrome_options.add_argument('--enable-unsafe-swiftshader')
            chrome_options.add_argument('--ignore-gpu-blocklist')
            chrome_options.add_argument('--enable-gpu-rasterization')
            
            # Network settings
            chrome_options.add_argument('--dns-prefetch-disable')
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-default-apps')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-sync')
            chrome_options.add_argument('--disable-translate')
            chrome_options.add_argument('--hide-scrollbars')
            chrome_options.add_argument('--metrics-recording-only')
            chrome_options.add_argument('--mute-audio')
            chrome_options.add_argument('--no-first-run')
            chrome_options.add_argument('--safebrowsing-disable-auto-update')
            
            # Random user agent
            ua = UserAgent()
            chrome_options.add_argument(f'--user-agent={ua.random}')
            
            # Additional preferences
            prefs = {
                'profile.default_content_setting_values': {
                    'notifications': 2,
                    'images': 2,
                    'javascript': 1,
                    'cookies': 1,
                    'plugins': 2,
                    'popups': 2,
                    'geolocation': 2,
                    'media_stream': 2,
                },
                'profile.managed_default_content_settings': {
                    'javascript': 1,
                    'cookies': 1,
                },
                'profile.cookie_controls_mode': 0,
                'credentials_enable_service': False,
                'profile.password_manager_enabled': False,
                'profile.default_content_settings.popups': 0,
                'profile.managed_default_content_settings.images': 2,
                'profile.default_content_setting_values.notifications': 2,
                'profile.managed_default_content_settings.javascript': 1,
                'profile.managed_default_content_settings.cookies': 1,
                'profile.managed_default_content_settings.plugins': 2,
                'profile.managed_default_content_settings.popups': 2,
                'profile.managed_default_content_settings.geolocation': 2,
                'profile.managed_default_content_settings.media_stream': 2,
            }
            chrome_options.add_experimental_option('prefs', prefs)
            
            # Add random viewport size
            viewport_width = random.randint(1024, 1920)
            viewport_height = random.randint(768, 1080)
            chrome_options.add_argument(f'--window-size={viewport_width},{viewport_height}')
            
            # Create undetected driver with increased timeout
            driver = uc.Chrome(
                options=chrome_options,
                driver_executable_path=None,  # Let it auto-download
                browser_executable_path=None,  # Let it auto-detect
                suppress_welcome=True,
                headless=False,
                use_subprocess=True,
                version_main=None,  # Auto-detect Chrome version
                timeout=30  # Increased timeout for driver creation
            )
            
            # Apply fingerprint
            self.fingerprint_manager.apply_fingerprint(driver)
            
            # Apply stealth settings
            stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform=platform.system(),
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            
            # Add random mouse movements
            self.human_behavior.random_mouse_movement(driver)
            
            return driver
        except Exception as e:
            logger.error(f"Failed to create undetected driver: {e}")
            raise
            
    def _add_random_mouse_movements(self, driver):
        """Add random mouse movements to appear more human-like"""
        try:
            action = ActionChains(driver)
            for _ in range(random.randint(3, 7)):
                x = random.randint(0, 1000)
                y = random.randint(0, 700)
                action.move_by_offset(x, y)
                action.pause(random.uniform(0.1, 0.3))
            action.perform()
        except:
            pass
            
    def _human_like_delay(self):
        """Add random delay to simulate human behavior"""
        time.sleep(random.uniform(1, 3))
        
    def _handle_captcha(self, driver):
        """Handle potential CAPTCHA challenges"""
        try:
            # Check for common CAPTCHA elements
            captcha_elements = driver.find_elements(By.XPATH, 
                "//*[contains(text(), 'captcha') or contains(@class, 'captcha')]")
            
            if captcha_elements:
                logger.warning("CAPTCHA detected, waiting for manual intervention")
                # Wait for manual intervention
                time.sleep(30)
                return True
            return False
        except:
            return False
        
    def fetch_economic_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch economic calendar events for a date range.
        """
        # Check cache first
        if self.mongo_client:
            cached_events = self._get_cached_events(start_date, end_date)
            if cached_events:
                return cached_events
        
        events = []
        
        try:
            # Scrape from Investing.com
            events = self.fetch_high_impact_us_events()
            
            # Store in MongoDB
            if events and self.mongo_client:
                self._store_events(events)
                
        except Exception as e:
            logger.error(f"Error fetching economic events: {e}")
            
        return events
    
    def _apply_filters(self, driver: WebDriver) -> bool:
        """Apply filters to show only US high-impact events"""
        try:
            # Wait for filter button and click it
            filter_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "filterStateAnchor"))
            )
            driver.execute_script("arguments[0].click();", filter_button)
            time.sleep(2)  # Wait for filter panel to open
            
            # Clear all countries first
            clear_all = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Clear')]"))
            )
            driver.execute_script("arguments[0].click();", clear_all)
            time.sleep(1)
            
            # Select only United States
            us_checkbox = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@id='country5']"))
            )
            if not us_checkbox.is_selected():
                driver.execute_script("arguments[0].click();", us_checkbox)
            time.sleep(1)
            
            # Select only high importance (3 stars)
            high_importance = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@id='importance3']"))
            )
            if not high_importance.is_selected():
                driver.execute_script("arguments[0].click();", high_importance)
                time.sleep(1)

            # Click Apply button
            apply_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "ecSubmitButton"))
            )
            driver.execute_script("arguments[0].click();", apply_button)
            time.sleep(3)  # Wait for filters to apply
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return False

    def fetch_high_impact_us_events(self) -> List[Dict]:
        """Fetch high-impact US economic events from Investing.com with instance-level caching"""
        
        # Check instance-level cache first
        if (self._events_cache['data'] is not None and 
            self._events_cache['timestamp'] is not None and
            (datetime.now() - self._events_cache['timestamp']) < self._events_cache['cache_duration']):
            
            logger.info("Using instance-cached economic events (no web scraping needed)")
            return self._events_cache['data']
        
        driver = None
        try:
            driver = self._create_undetected_driver()
            wait = WebDriverWait(driver, 10)
            
            # Set desktop user agent explicitly
            desktop_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": desktop_user_agent})

            # Navigate to economic calendar
            url = "https://www.investing.com/economic-calendar/"
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    # Wait for page to be fully loaded
                    wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Page load attempt {attempt + 1} failed: {e}")
                    time.sleep(5)  # Wait before retry
            
            # Add cookies
            cookies = [
                {'name': 'cookieConsent', 'value': '1'},
                {'name': 'cookieConsentLevel', 'value': 'all'},
                {'name': 'cookieConsentType', 'value': 'all'}
            ]
            for cookie in cookies:
                driver.add_cookie(cookie)
            
            # Refresh page to apply cookies
            driver.refresh()
            time.sleep(2)
            
            # Click "This Week" button with retry
            for attempt in range(max_retries):
                try:
                    this_week_button = wait.until(
                        EC.element_to_be_clickable((By.ID, "timeFrame_thisWeek"))
                    )
                    driver.execute_script("arguments[0].click();", this_week_button)
                    time.sleep(2)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"This Week button click attempt {attempt + 1} failed: {e}")
                    time.sleep(5)
            
            # Apply filters
            if not self._apply_filters(driver):
                raise Exception("Failed to apply filters")
            
            # Wait for events to load
            time.sleep(3)
            
            # Get all event rows
            event_rows = []
            selectors = [
                "tr.js-event-item",
                "tr.eventRow",
                "tr[data-event-datetime]"
            ]
            
            for selector in selectors:
                try:
                    rows = driver.find_elements(By.CSS_SELECTOR, selector)
                    if rows:
                        event_rows = rows
                        break
                except Exception as e:
                    logger.warning(f"Failed to find events with selector {selector}: {e}")
                    continue

            if not event_rows:
                logger.warning("No event rows found with any selector")
                return []

            # Parse events
            events = []
            for row in event_rows:
                try:
                    event_data = self._parse_event_row(row)
                    if event_data:
                        events.append(event_data)
                except Exception as e:
                    logger.warning(f"Failed to parse event row: {e}")
                    continue

            # Cache the fetched events for 6 hours
            self._events_cache['data'] = events
            self._events_cache['timestamp'] = datetime.now()
            logger.info(f"Cached {len(events)} economic events for 6 hours")

            return events
        except Exception as e:
            logger.error(f"Error fetching economic events: {e}")
            return []
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _parse_event_row(self, row) -> Optional[Dict]:
        try:
            event_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),  # Default to current date
                'time': None,
                'event': None,
                'importance': 'high',  # since we filter for high impact
                'actual': None,
                'forecast': None,
                'previous': None,
                'country': 'US',
                'affected_tickers': [],
                'source': 'investing.com'
            }
            # Extract event name
            try:
                event_elem = row.find_element(By.CSS_SELECTOR, "td.event")
                event_data['event'] = event_elem.text.strip()
            except:
                pass
            # Extract date
            try:
                date_elem = row.find_element(By.CSS_SELECTOR, "td.date")
                event_data['date'] = date_elem.text.strip()
            except:
                pass
            # Extract time
            try:
                time_elem = row.find_element(By.CSS_SELECTOR, "td.time")
                event_data['time'] = time_elem.text.strip()
            except:
                pass
            # Extract actual, forecast, previous
            for key, selector in [('actual', 'td.act'), ('forecast', 'td.fore'), ('previous', 'td.prev')]:
                try:
                    elem = row.find_element(By.CSS_SELECTOR, selector)
                    event_data[key] = elem.text.strip()
                except:
                    pass
            # Extract country (ensure 'US')
            try:
                flag_elem = row.find_element(By.CSS_SELECTOR, "td.flagCur")
                country_class = flag_elem.get_attribute("class")
                if 'USA' in country_class or 'US' in country_class:
                    event_data['country'] = 'US'
            except:
                pass
            # Extract affected tickers
            event_data['affected_tickers'] = self._map_event_to_tickers(event_data)
            return event_data
        except Exception as e:
            logger.error(f"Error parsing event row: {e}")
            return None
    
    def _parse_importance(self, bull_count: int) -> str:
        """Convert bull count to importance level."""
        if bull_count >= 3:
            return 'high'
        elif bull_count == 2:
            return 'medium'
        else:
            return 'low'
    
    def _parse_event_date(self, date_str: str) -> datetime:
        """Parse event date string to datetime."""
        try:
            # Handle various date formats
            if '/' in date_str:
                return datetime.strptime(date_str, '%m/%d/%Y')
            elif '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # Default to today if parsing fails
                return datetime.now()
        except:
            return datetime.now()
    
    def _extract_country_from_class(self, class_str: str) -> str:
        """Extract country code from CSS class."""
        # Class format: "ceFlags USA" or similar
        parts = class_str.split()
        if len(parts) > 1:
            country = parts[1]
            # Map common country names to codes
            country_map = {
                'USA': 'US',
                'United_States': 'US',
                'UnitedStates': 'US',
                'United_States_of_America': 'US'
            }
            return country_map.get(country, country)
        return 'US'  # Default to US if no country found
    
    def _map_event_to_tickers(self, event_data: Dict) -> List[str]:
        """
        Map economic event to affected tickers based on event type.
        """
        affected_tickers = set()
        event_name = event_data.get('event', '').lower()
        
        # Check each mapping keyword
        for keyword, tickers in EVENT_TICKER_MAPPING.items():
            if keyword in event_name:
                affected_tickers.update(tickers)
        
        # High importance events affect broad market
        if event_data.get('importance') == 'high' and not affected_tickers:
            affected_tickers.update(['SPY', 'QQQ', 'IWM', 'DIA'])
        
        # Filter to only S&P 100 tickers if needed
        from ..config.constants import TOP_100_TICKERS
        affected_tickers = [t for t in affected_tickers if t in TOP_100_TICKERS]
        
        return list(affected_tickers)
    
    def get_event_features(self, ticker: str, date: datetime, lookback_days: int = 7) -> Dict:
        """Get event-driven features for a specific ticker and date."""
        features = {
            'has_high_impact_event_today': 0,
            'days_to_next_high_impact': 30,  # Changed from 999 to 30 (more realistic default)
            'days_since_last_high_impact': 30,  # Changed from 999 to 30
            'event_density_7d': 0,
            'event_importance_sum_7d': 0.0,
            'has_earnings_today': 0,
            'days_to_next_earnings': 90,  # Changed from 999 to 90 (quarterly earnings)
            'days_since_last_earnings': 90,  # Changed from 999 to 90
            'has_dividend_today': 0,
            'days_to_next_dividend': 90,  # Changed from 999 to 90 (quarterly dividends)
            'days_since_last_dividend': 90,  # Changed from 999 to 90
            'dividend_amount': 0.0,
            'dividend_yield': 0.0,
            'next_earnings_eps_estimate': None,
            'next_earnings_revenue_estimate': None,
            'next_dividend_record_date': None,
            'next_dividend_payment_date': None,
            'dividend_frequency': None,
            'data_sources_checked': [],
            'fmp_earnings_count': 0,
            'fmp_dividends_count': 0,
            'economic_events_count': 0,
            'sentiment_data_available': False
        }
        
        try:
            # Get economic events
            start_date = date - timedelta(days=lookback_days)
            end_date = date + timedelta(days=lookback_days)
            events = self.fetch_economic_events(start_date, end_date)
            
            # Get latest sentiment data from MongoDB which includes FMP data
            features['data_sources_checked'].append('sentiment_data')
            if self.mongo_client and self.mongo_client.db is not None:
                sentiment_data = self.mongo_client.get_latest_sentiment(ticker)
                if sentiment_data:
                    features['sentiment_data_available'] = True
                    logger.info(f"✓ Found sentiment data for {ticker} with keys: {list(sentiment_data.keys())}")
                    
                    # Check multiple locations for FMP data
                    fmp_data_sources = [
                        sentiment_data.get('fmp_raw_data', {}),  # Primary location
                        sentiment_data,  # Direct in sentiment data
                    ]
                    
                    fmp_data = {}
                    for source in fmp_data_sources:
                        if isinstance(source, dict) and source:
                            fmp_data = source
                            break
                    
                    if fmp_data:
                        logger.info(f"Found FMP data for {ticker} with keys: {list(fmp_data.keys())}")
                        
                        # Check earnings data using multiple possible locations
                        earnings_data = []
                        
                        # Try different possible locations for earnings data
                        earnings_sources = [
                            fmp_data.get('earnings', []),           # Direct earnings array
                            fmp_data.get('earnings_calendar', []),  # Calendar-specific earnings
                            sentiment_data.get('fmp_earnings', []), # Legacy FMP location
                            sentiment_data.get('fmp_earnings_calendar', [])  # Legacy location
                        ]
                        
                        for source in earnings_sources:
                            if source and isinstance(source, list) and len(source) > 0:
                                earnings_data = source
                                logger.info(f"Found earnings data from source with {len(source)} entries")
                                break
                                
                        if earnings_data:
                            features['fmp_earnings_count'] = len(earnings_data)
                            features['data_sources_checked'].append('fmp_earnings')
                            logger.info(f"Processing {len(earnings_data)} FMP earnings entries for {ticker}")
                            
                            for event in earnings_data:
                                try:
                                    # Handle both FMP API structures
                                    event_date_str = event.get('date') or event.get('reportedDate')
                                    if not event_date_str:
                                        continue
                                        
                                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                                    days_diff = (event_date - date).days
                                    
                                    logger.info(f"Processing earnings event: {event_date_str}, days_diff: {days_diff}")
                                    
                                    if days_diff == 0:
                                        features['has_earnings_today'] = 1
                                        features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                                        features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                                        logger.info(f"✓ Earnings TODAY for {ticker}: EPS={features['next_earnings_eps_estimate']}, Revenue={features['next_earnings_revenue_estimate']}")
                                    elif days_diff > 0 and days_diff < features['days_to_next_earnings']:
                                        features['days_to_next_earnings'] = days_diff
                                        features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                                        features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                                        logger.info(f"✓ Next earnings for {ticker} in {days_diff} days: EPS={features['next_earnings_eps_estimate']}, Revenue={features['next_earnings_revenue_estimate']}")
                                    elif days_diff < 0 and abs(days_diff) < features['days_since_last_earnings']:
                                        features['days_since_last_earnings'] = abs(days_diff)
                                        logger.info(f"✓ Last earnings for {ticker} was {abs(days_diff)} days ago: EPS_Actual={event.get('epsActual')}")
                                except Exception as e:
                                    logger.warning(f"Error processing earnings event for {ticker}: {e}")
                        else:
                            logger.warning(f"No earnings data found in any location for {ticker}")
                        
                        # Check dividends data using multiple possible locations
                        dividends_data = []
                        
                        # Try different possible locations for dividends data
                        dividends_sources = [
                            fmp_data.get('dividends', []),           # Direct dividends array
                            fmp_data.get('dividends_calendar', []),  # Calendar-specific dividends
                            sentiment_data.get('fmp_dividends', []), # Legacy FMP location
                            sentiment_data.get('fmp_dividends_calendar', [])  # Legacy location
                        ]
                        
                        for source in dividends_sources:
                            if source and isinstance(source, list) and len(source) > 0:
                                dividends_data = source
                                logger.info(f"Found dividends data from source with {len(source)} entries")
                                break
                                
                        if dividends_data:
                            features['fmp_dividends_count'] = len(dividends_data)
                            features['data_sources_checked'].append('fmp_dividends')
                            logger.info(f"Processing {len(dividends_data)} FMP dividends entries for {ticker}")
                            
                            for div in dividends_data:
                                try:
                                    # Handle both FMP API structures
                                    div_date_str = div.get('date') or div.get('exDividendDate')  # Ex-dividend date
                                    if not div_date_str:
                                        continue
                                        
                                    div_date = datetime.strptime(div_date_str, '%Y-%m-%d')
                                    days_diff = (div_date - date).days
                                    
                                    logger.info(f"Processing dividend event: {div_date_str}, days_diff: {days_diff}")
                                    
                                    if days_diff == 0:
                                        features['has_dividend_today'] = 1
                                        features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                                        features['dividend_yield'] = float(div.get('yield', 0))
                                        features['next_dividend_record_date'] = div.get('recordDate')
                                        features['next_dividend_payment_date'] = div.get('paymentDate')
                                        features['dividend_frequency'] = div.get('frequency')
                                        logger.info(f"✓ Dividend TODAY for {ticker}: ${features['dividend_amount']:.3f} ({features['dividend_yield']:.2f}% yield)")
                                    elif days_diff > 0 and days_diff < features['days_to_next_dividend']:
                                        features['days_to_next_dividend'] = days_diff
                                        features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                                        features['dividend_yield'] = float(div.get('yield', 0))
                                        features['next_dividend_record_date'] = div.get('recordDate')
                                        features['next_dividend_payment_date'] = div.get('paymentDate')
                                        features['dividend_frequency'] = div.get('frequency')
                                        logger.info(f"✓ Next dividend for {ticker} in {days_diff} days: ${features['dividend_amount']:.3f} ({features['dividend_yield']:.2f}% yield)")
                                    elif days_diff < 0 and abs(days_diff) < features['days_since_last_dividend']:
                                        features['days_since_last_dividend'] = abs(days_diff)
                                        logger.info(f"✓ Last dividend for {ticker} was {abs(days_diff)} days ago: ${div.get('dividend', 0):.3f}")
                                except Exception as e:
                                    logger.warning(f"Error processing dividend event for {ticker}: {e}")
                        else:
                            logger.warning(f"No dividends data found in any location for {ticker}")
                                    
                    else:
                        logger.warning(f"No FMP data found in sentiment data for {ticker}")
                        
                    # Log final feature values for debugging
                    logger.info(f"Final economic features for {ticker}:")
                    logger.info(f"  - FMP earnings count: {features['fmp_earnings_count']}")
                    logger.info(f"  - FMP dividends count: {features['fmp_dividends_count']}")
                    logger.info(f"  - Days to next earnings: {features['days_to_next_earnings']}")
                    logger.info(f"  - Days to next dividend: {features['days_to_next_dividend']}")
                    logger.info(f"  - Data sources checked: {features['data_sources_checked']}")
                else:
                    features['sentiment_data_available'] = False
                    logger.warning(f"No sentiment data found for {ticker}")

            # Process economic events - only high impact events
            features['economic_events_count'] = len(events)
            features['data_sources_checked'].append('economic_events')
            logger.info(f"Found {len(events)} economic events in date range for processing")
            
            ticker_events = [e for e in events if ticker in e.get('affected_tickers', [])]
            logger.info(f"Found {len(ticker_events)} events specifically affecting {ticker}")
            
            for event in ticker_events:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                days_diff = (event_date - date).days
                
                if days_diff == 0:
                        features['has_high_impact_event_today'] = 1
                
                if days_diff > 0 and days_diff <= 7:
                    features['event_density_7d'] += 1
                    features['event_importance_sum_7d'] += 1.0  # All events are high impact
                    
                    if days_diff < features['days_to_next_high_impact']:
                        features['days_to_next_high_impact'] = days_diff
                
                if days_diff < 0:
                    days_since = abs(days_diff)
                    if days_since < features['days_since_last_high_impact']:
                        features['days_since_last_high_impact'] = days_since
            
            # Add event volatility score
            features['event_volatility_score'] = self._calculate_event_volatility(features)
            
        except Exception as e:
            logger.error(f"Error getting event features for {ticker}: {e}")
        
        return features
    
    def _calculate_event_volatility(self, features: Dict) -> float:
        """Calculate expected volatility based on economic events and corporate events."""
        score = 0.0
        
        # Economic events impact
        score += features['has_high_impact_event_today'] * 1.0
        
        # Earnings impact
        if features['has_earnings_today']:
            score += 1.2  # Earnings have higher impact than economic events
        elif features['days_to_next_earnings'] <= 2:
            score += 0.6
        elif features['days_to_next_earnings'] <= 5:
            score += 0.3
        
        # Dividend impact
        if features['has_dividend_today']:
            score += 0.4  # Dividends have moderate impact
        elif features['days_to_next_dividend'] <= 2:
            score += 0.2
            
        # Event clustering increases volatility
        if features['event_density_7d'] > 5:
            score += 0.4
        elif features['event_density_7d'] > 3:
            score += 0.2
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _calculate_real_earnings_days(self, fmp_data: Dict, current_date: datetime) -> Dict:
        """Calculate real days to/from earnings using FMP data."""
        earnings_features = {
            'days_to_next_earnings': 365,
            'days_since_last_earnings': 365,
            'fmp_earnings_count': 0,
            'next_earnings_eps_estimate': None,
            'next_earnings_revenue_estimate': None,
            'has_earnings_today': 0
        }
        
        try:
            # Combine earnings calendar and historical earnings
            earnings_calendar = fmp_data.get('earnings_calendar', [])
            historical_earnings = fmp_data.get('historical_earnings', [])
            
            all_earnings = earnings_calendar + historical_earnings
            if not all_earnings:
                return earnings_features
            
            earnings_features['fmp_earnings_count'] = len(all_earnings)
            
            for event in all_earnings:
                try:
                    event_date_str = event.get('date') or event.get('reportedDate')
                    if not event_date_str:
                        continue
                        
                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                    days_diff = (event_date - current_date).days
                    
                    if days_diff == 0:
                        earnings_features['has_earnings_today'] = 1
                        earnings_features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                        earnings_features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                    elif days_diff > 0 and days_diff < earnings_features['days_to_next_earnings']:
                        earnings_features['days_to_next_earnings'] = days_diff
                        earnings_features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                        earnings_features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                    elif days_diff < 0 and abs(days_diff) < earnings_features['days_since_last_earnings']:
                        earnings_features['days_since_last_earnings'] = abs(days_diff)
                        
                except Exception as e:
                    logger.warning(f"Error processing earnings event: {e}")
                    
            return earnings_features
            
        except Exception as e:
            logger.error(f"Error calculating real earnings days: {e}")
            return earnings_features
    
    def _calculate_real_dividend_days(self, fmp_data: Dict, current_date: datetime) -> Dict:
        """Calculate real days to/from dividends using FMP data."""
        dividend_features = {
            'days_to_next_dividend': 365,
            'days_since_last_dividend': 365,
            'fmp_dividends_count': 0,
            'dividend_amount': None,
            'dividend_yield': None,
            'next_dividend_record_date': None,
            'next_dividend_payment_date': None,
            'dividend_frequency': None,
            'has_dividend_today': 0
        }
        
        try:
            # Combine dividends calendar and historical dividends
            dividends_calendar = fmp_data.get('dividends_calendar', [])
            historical_dividends = fmp_data.get('historical_dividends', [])
            
            all_dividends = dividends_calendar + historical_dividends
            if not all_dividends:
                return dividend_features
            
            dividend_features['fmp_dividends_count'] = len(all_dividends)
            
            for div in all_dividends:
                try:
                    div_date_str = div.get('date') or div.get('exDividendDate') or div.get('paymentDate')
                    if not div_date_str:
                        continue
                        
                    div_date = datetime.strptime(div_date_str, '%Y-%m-%d')
                    days_diff = (div_date - current_date).days
                    
                    if days_diff == 0:
                        dividend_features['has_dividend_today'] = 1
                        dividend_features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                        dividend_features['dividend_yield'] = float(div.get('yield', 0))
                        dividend_features['next_dividend_record_date'] = div.get('recordDate')
                        dividend_features['next_dividend_payment_date'] = div.get('paymentDate')
                        dividend_features['dividend_frequency'] = div.get('frequency')
                    elif days_diff > 0 and days_diff < dividend_features['days_to_next_dividend']:
                        dividend_features['days_to_next_dividend'] = days_diff
                        dividend_features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                        dividend_features['dividend_yield'] = float(div.get('yield', 0))
                        dividend_features['next_dividend_record_date'] = div.get('recordDate')
                        dividend_features['next_dividend_payment_date'] = div.get('paymentDate')
                        dividend_features['dividend_frequency'] = div.get('frequency')
                    elif days_diff < 0 and abs(days_diff) < dividend_features['days_since_last_dividend']:
                        dividend_features['days_since_last_dividend'] = abs(days_diff)
                        
                except Exception as e:
                    logger.warning(f"Error processing dividend event: {e}")
                    
            return dividend_features
            
        except Exception as e:
            logger.error(f"Error calculating real dividend days: {e}")
            return dividend_features
    
    def _get_cached_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get cached events from MongoDB"""
        try:
            if not self.mongo_client or self.mongo_client.db is None:
                return []
            
            # Query MongoDB for events in date range
            query = {
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            events = list(self.mongo_client.db.economic_events.find(query))
            if events:
                logger.info(f"Found {len(events)} cached events")
            return events
            
        except Exception as e:
            logger.error(f"Error getting cached events: {e}")
            return []
    
    def _store_events(self, events: List[Dict]):
        """Store events in MongoDB with duplicate prevention"""
        try:
            if not self.mongo_client or self.mongo_client.db is None or not events:
                return
            
            stored_count = 0
            for event in events:
                # Use upsert to prevent duplicates based on date and event name
                unique_key = {
                    'date': event.get('date'),
                    'event': event.get('event'),
                    'country': event.get('country', 'US')
                }
                
                result = self.mongo_client.db.economic_events.replace_one(
                    unique_key,
                    event,
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    stored_count += 1
            
            logger.info(f"Stored/Updated {stored_count} unique events in MongoDB (processed {len(events)} total)")
            
        except Exception as e:
            logger.error(f"Error storing events: {e}")

    async def fetch_fresh_fmp_data(self, ticker: str) -> Dict:
        """
        Fetch fresh FMP earnings and dividend data using centralized FMP manager.
        This prevents duplicate API calls by using the same manager as sentiment.py.
        """
        try:
            # Import the FMP manager from sentiment.py to avoid duplicate API calls
            from .sentiment import FMPAPIManager
            
            # Create FMP manager instance (this will use caching)
            fmp_manager = FMPAPIManager(self.mongo_client)
            
            logger.info(f"Fetching fresh FMP data for {ticker} using centralized manager")
            
            # Get all FMP data in one consolidated call
            fresh_data = await fmp_manager.get_all_fmp_data(ticker)
            
            if not fresh_data:
                logger.warning(f"No FMP data returned for {ticker}")
                return {}
            
            logger.info(f"Successfully fetched FMP data for {ticker}: {list(fresh_data.keys())}")
            return fresh_data
            
        except Exception as e:
            logger.error(f"Error fetching fresh FMP data for {ticker}: {e}")
            return {}

    async def get_event_features_with_fresh_data(self, ticker: str, date: datetime, lookback_days: int = 7) -> Dict:
        """
        Get event features using fresh FMP data instead of placeholders.
        """
        try:
            # Get fresh FMP data (this uses centralized manager to avoid duplicate API calls)
            fresh_fmp_data = await self.fetch_fresh_fmp_data(ticker)
            
            # Start with the standard event features
            features = self.get_event_features(ticker, date, lookback_days)
            
            if fresh_fmp_data:
                # Update features with real FMP data
                features.update(self._calculate_real_earnings_days(fresh_fmp_data, date))
                features.update(self._calculate_real_dividend_days(fresh_fmp_data, date))
                
                logger.info(f"Updated economic features for {ticker} with fresh FMP data")
            else:
                logger.warning(f"No fresh FMP data available for {ticker}, using default features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting event features with fresh data for {ticker}: {e}")
            # Fallback to standard features
            return self.get_event_features(ticker, date, lookback_days)

def integrate_economic_events_sentiment(sentiment_dict: Dict, ticker: str, mongo_client=None) -> Dict:
    """
    Integrate economic event data into sentiment analysis.
    High-impact events increase uncertainty and potential volatility.
    Uses shared EconomicCalendar instance when possible to prevent multiple web scraping.
    """
    try:
        # Try to use shared calendar instance from sentiment analyzer
        from .sentiment import SHARED_ECONOMIC_CALENDAR
        if SHARED_ECONOMIC_CALENDAR is not None:
            calendar = SHARED_ECONOMIC_CALENDAR
        else:
            # Fallback to creating new instance
            calendar = EconomicCalendar(mongo_client)
        
        # Get event features for today
        event_features = calendar.get_event_features(ticker, datetime.now())
        
        # Calculate event-based sentiment adjustment
        event_sentiment = 0.0
        
        # High impact events today create uncertainty (slightly bearish)
        if event_features['has_high_impact_event_today']:
            event_sentiment -= 0.2
        
        # Many upcoming events create anticipation (neutral to slightly bearish)
        if event_features['event_density_7d'] > 5:
            event_sentiment -= 0.1
        
        # Recent high impact events may still affect sentiment
        if event_features['days_since_last_high_impact'] <= 2:
            event_sentiment -= 0.05
        
        # High volatility score indicates trading opportunity (can be positive)
        volatility_score = event_features['event_volatility_score']
        if volatility_score > 1.5:
            event_sentiment += 0.1  # High vol can mean opportunity
        
        # Add to sentiment dict
        sentiment_dict['economic_event_sentiment'] = event_sentiment
        sentiment_dict['economic_event_volatility'] = volatility_score
        sentiment_dict['economic_event_volume'] = 1 if event_features['has_high_impact_event_today'] else 0
        sentiment_dict['economic_event_confidence'] = 0.9  # High confidence in event data
        
        # Store event features for explainability
        sentiment_dict['economic_event_features'] = event_features
        
        logger.info(f"Economic event sentiment for {ticker}: {event_sentiment:.2f} (volatility: {volatility_score:.2f})")
        
    except Exception as e:
        logger.error(f"Error integrating economic events: {e}")
        sentiment_dict['economic_event_sentiment'] = 0.0
        sentiment_dict['economic_event_volume'] = 0
        sentiment_dict['economic_event_confidence'] = 0.0
    
    return sentiment_dict 