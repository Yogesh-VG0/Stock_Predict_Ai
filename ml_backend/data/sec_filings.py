"""
Module for analyzing SEC filings using Kaleidoscope API (primary) and FMP (backup).
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
from ..utils.mongodb import MongoDBClient
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# transformers/torch removed — VADER is used for SEC sentiment analysis
# (transformers adds ~2 GB and fails to install in GitHub Actions CI)
import aiohttp
import statistics

# Load environment variables from various possible locations
import os.path
possible_env_paths = [
    '.env',
    '../.env', 
    '../../.env',
    'ml_backend/.env',
    os.path.join(os.path.dirname(__file__), '../.env'),
    os.path.join(os.path.dirname(__file__), '../../.env')
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

if not env_loaded:
    load_dotenv()  # Try default location anyway

logger = logging.getLogger(__name__)



class BaseSECFormParser:
    """Base class for SEC form parsers."""
    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "html.parser")
        
    def _get_form_data(self, label: str) -> str:
        """Helper method to get form data by label."""
        try:
            element = self.soup.find(text=lambda t: t and label in t)
            if element:
                return element.find_next("td", class_="FormData").text.strip()
            return ""
        except Exception:
            return ""
            
    def _get_form_text(self, label: str) -> str:
        """Helper method to get form text by label."""
        try:
            element = self.soup.find(text=lambda t: t and label in t)
            if element:
                return element.find_next("td", class_="FormText").text.strip()
            return ""
        except Exception:
            return ""
            
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis. Override in subclasses."""
        return ""

class Form25Parser(BaseSECFormParser):
    """Parser for Form 25 (Notification of removal from listing)."""
    def parse(self) -> dict:
        data = {
            'form_type': '25',
            'issuer': self._get_form_data("Issuer:"),
            'exchange': self._get_form_data("Exchange:"),
            'address': self._get_form_data("Address:"),
            'date': self._get_form_data("Date:"),
            'securities_removed': self._get_form_data("Securities:"),
            'rule_provisions': self._get_form_data("Rule:"),
            'signatory': self._get_form_data("By:")
        }
        return data

class Form4Parser(BaseSECFormParser):
    """Parser for Form 4 (Statement of changes in beneficial ownership)."""
    def parse(self) -> dict:
        data = {
            'form_type': '4',
            'reporting_person': self._get_form_data("Reporting Person:"),
            'issuer': self._get_form_data("Issuer:"),
            'title': self._get_form_data("Title:"),
            'ownership_type': self._get_form_data("Ownership Type:"),
            'transaction_date': self._get_form_data("Transaction Date:"),
            'transaction_type': self._get_form_data("Transaction Type:"),
            'amount': self._get_form_data("Amount:"),
            'price': self._get_form_data("Price:"),
            'officer_role': self._get_form_data("Officer Role:")
        }
        
        # Parse transaction table if present
        try:
            table = self.soup.find("table", class_="FormData")
            if table:
                transactions = []
                for row in table.find_all("tr")[1:]:  # Skip header row
                    cols = row.find_all("td")
                    if len(cols) >= 4:
                        transactions.append({
                            'security': cols[0].text.strip(),
                            'transaction_type': cols[1].text.strip(),
                            'amount': cols[2].text.strip(),
                            'price': cols[3].text.strip()
                        })
                data['transactions'] = transactions
        except Exception as e:
            logger.error(f"Error parsing Form 4 transactions: {str(e)}")
            
        return data

class Form8KParser(BaseSECFormParser):
    """Parser for Form 8-K (Current report)."""
    def parse(self) -> dict:
        data = {
            'form_type': '8-K',
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'commission_file': self._get_form_data("Commission File:"),
            'state': self._get_form_data("State:"),
            'address': self._get_form_data("Address:")
        }
        
        # Parse events and disclosure checkboxes
        try:
            events = []
            event_sections = self.soup.find_all("div", class_="FormGroup")
            for section in event_sections:
                event = {
                    'description': self._get_form_text("Event Description:"),
                    'date': self._get_form_data("Event Date:"),
                    'disclosures': []
                }
                
                # Get disclosure checkboxes
                checkboxes = section.find_all("input", type="checkbox")
                for cb in checkboxes:
                    if cb.get('checked'):
                        event['disclosures'].append(cb.find_next("label").text.strip())
                        
                events.append(event)
            data['events'] = events
        except Exception as e:
            logger.error(f"Error parsing Form 8-K events: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = []
        event_sections = self.soup.find_all("div", class_="FormGroup")
        for section in event_sections:
            description = self._get_form_text("Event Description:")
            if description:
                sections.append(description)
        return " ".join(sections)

class Form10KParser(BaseSECFormParser):
    """Parser for Form 10-K (Annual report)."""
    def parse(self) -> dict:
        data = {
            'form_type': '10-K',
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'period': self._get_form_data("Period:"),
            'business_overview': self._get_form_text("Business Overview:"),
            'risk_factors': self._get_form_text("Risk Factors:"),
            'financial_statements': {}
        }
        
        # Parse XBRL financial data if present
        try:
            xbrl_tags = self.soup.find_all("xbrl:tag")
            for tag in xbrl_tags:
                name = tag.get('name', '')
                value = tag.text.strip()
                if name and value:
                    data['financial_statements'][name] = value
        except Exception as e:
            logger.error(f"Error parsing Form 10-K XBRL data: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = [
            self._get_form_text("Business Overview:"),
            self._get_form_text("Risk Factors:"),
            self._get_form_text("Management Discussion:"),
            self._get_form_text("Financial Discussion:")
        ]
        return " ".join(filter(None, sections))

class Form10QParser(BaseSECFormParser):
    """Parser for Form 10-Q (Quarterly report)."""
    def parse(self) -> dict:
        data = {
            'form_type': '10-Q',
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'period': self._get_form_data("Period:"),
            'company_info': self._get_form_text("Company Information:"),
            'financial_discussion': self._get_form_text("Financial Discussion:"),
            'financial_statements': {}
        }
        
        # Parse XBRL financial data if present
        try:
            xbrl_tags = self.soup.find_all("xbrl:tag")
            for tag in xbrl_tags:
                name = tag.get('name', '')
                value = tag.text.strip()
                if name and value:
                    data['financial_statements'][name] = value
        except Exception as e:
            logger.error(f"Error parsing Form 10-Q XBRL data: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = [
            self._get_form_text("Company Information:"),
            self._get_form_text("Financial Discussion:"),
            self._get_form_text("Management Discussion:")
        ]
        return " ".join(filter(None, sections))

class Form144Parser(BaseSECFormParser):
    """Parser for Form 144 (Notice of proposed sale of securities)."""
    def parse(self) -> dict:
        data = {
            'form_type': '144',
            'issuer': self._get_form_data("Issuer:"),
            'insider_name': self._get_form_data("Reporting Person:"),
            'securities_sold': self._get_form_data("Securities:"),
            'amount': self._get_form_data("Amount:"),
            'price': self._get_form_data("Price:"),
            'date': self._get_form_data("Date:"),
            'ownership': self._get_form_data("Ownership:"),
            'broker': self._get_form_data("Broker:"),
            'broker_address': self._get_form_data("Broker Address:")
        }
        
        # Parse additional ownership details if present
        try:
            ownership_table = self.soup.find("table", class_="FormData")
            if ownership_table:
                ownership_details = []
                for row in ownership_table.find_all("tr")[1:]:  # Skip header row
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        ownership_details.append({
                            'security_type': cols[0].text.strip(),
                            'amount': cols[1].text.strip(),
                            'ownership_type': cols[2].text.strip()
                        })
                data['ownership_details'] = ownership_details
        except Exception as e:
            logger.error(f"Error parsing Form 144 ownership details: {str(e)}")
            
        return data

class FormDEF14AParser(BaseSECFormParser):
    """Parser for Form DEF 14A (Proxy statement)."""
    def parse(self) -> dict:
        data = {
            'form_type': 'DEF 14A',
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'meeting_date': self._get_form_data("Meeting Date:"),
            'meeting_type': self._get_form_data("Meeting Type:"),
            'directors': [],
            'voting_matters': []
        }
        
        # Parse director information
        try:
            director_sections = self.soup.find_all("div", class_="DirectorInfo")
            for section in director_sections:
                director = {
                    'name': self._get_form_data("Name:"),
                    'age': self._get_form_data("Age:"),
                    'biography': self._get_form_text("Biography:"),
                    'compensation': self._get_form_text("Compensation:")
                }
                data['directors'].append(director)
        except Exception as e:
            logger.error(f"Error parsing Form DEF 14A directors: {str(e)}")
            
        # Parse voting matters
        try:
            voting_sections = self.soup.find_all("div", class_="VotingMatter")
            for section in voting_sections:
                matter = {
                    'description': self._get_form_text("Description:"),
                    'recommendation': self._get_form_text("Recommendation:"),
                    'vote_required': self._get_form_data("Vote Required:")
                }
                data['voting_matters'].append(matter)
        except Exception as e:
            logger.error(f"Error parsing Form DEF 14A voting matters: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = []
        
        # Add director biographies and compensation
        director_sections = self.soup.find_all("div", class_="DirectorInfo")
        for section in director_sections:
            sections.extend([
                self._get_form_text("Biography:"),
                self._get_form_text("Compensation:")
            ])
            
        # Add voting matters
        voting_sections = self.soup.find_all("div", class_="VotingMatter")
        for section in voting_sections:
            sections.extend([
                self._get_form_text("Description:"),
                self._get_form_text("Recommendation:")
            ])
            
        return " ".join(filter(None, sections))

class FormSDParser(BaseSECFormParser):
    """Parser for Form SD (Conflict minerals)."""
    def parse(self) -> dict:
        data = {
            'form_type': 'SD',
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'due_diligence': self._get_form_text("Due Diligence:"),
            'declaration': self._get_form_text("Declaration:"),
            'countries': []
        }
        
        # Parse country of origin information
        try:
            country_sections = self.soup.find_all("div", class_="CountryInfo")
            for section in country_sections:
                country = {
                    'name': self._get_form_data("Country:"),
                    'minerals': self._get_form_text("Minerals:"),
                    'smelters': self._get_form_text("Smelters:")
                }
                data['countries'].append(country)
        except Exception as e:
            logger.error(f"Error parsing Form SD countries: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = [
            self._get_form_text("Due Diligence:"),
            self._get_form_text("Declaration:")
        ]
        
        # Add country-specific information
        country_sections = self.soup.find_all("div", class_="CountryInfo")
        for section in country_sections:
            sections.extend([
                self._get_form_text("Minerals:"),
                self._get_form_text("Smelters:")
            ])
            
        return " ".join(filter(None, sections))

class Form11KParser(BaseSECFormParser):
    """Parser for Form 11-K (Employee stock plans)."""
    def parse(self) -> dict:
        data = {
            'form_type': '11-K',
            'plan_name': self._get_form_data("Plan Name:"),
            'registrant': self._get_form_data("Registrant:"),
            'filing_date': self._get_form_data("Filing Date:"),
            'auditor': self._get_form_data("Auditor:"),
            'financial_statements': {}
        }
        
        # Parse financial statements
        try:
            financial_sections = self.soup.find_all("div", class_="FinancialStatement")
            for section in financial_sections:
                statement = {
                    'type': self._get_form_data("Statement Type:"),
                    'period': self._get_form_data("Period:"),
                    'content': self._get_form_text("Content:")
                }
                data['financial_statements'][statement['type']] = statement
        except Exception as e:
            logger.error(f"Error parsing Form 11-K financial statements: {str(e)}")
            
        return data
        
    def get_sentiment_text(self) -> str:
        """Get text content for sentiment analysis."""
        sections = []
        
        # Add auditor information
        sections.append(self._get_form_text("Auditor Report:"))
        
        # Add financial statement content
        financial_sections = self.soup.find_all("div", class_="FinancialStatement")
        for section in financial_sections:
            sections.append(self._get_form_text("Content:"))
            
        return " ".join(filter(None, sections))

class SECFilingsAnalyzer:
    def __init__(self, mongo_client: MongoDBClient = None):
        """Initialize SEC Filings Analyzer."""
        self.kaleidoscope_api_key = os.getenv("KALEIDOSCOPE_API_KEY")
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://api.kscope.io/v2/sec/search"
        self.mongo_client = mongo_client
        self.vader = SentimentIntensityAnalyzer()
        self.max_filings = 30  # Maximum number of filings to fetch per stock
        
        # Log API key status
        logger.info(f"SEC Filings Analyzer initialized:")
        logger.info(f"  - Kaleidoscope API Key: {'✓ Available' if self.kaleidoscope_api_key else '✗ Missing'}")
        logger.info(f"  - FMP API Key: {'✓ Available' if self.fmp_api_key else '✗ Missing'}")
        
        if not self.kaleidoscope_api_key and not self.fmp_api_key:
            logger.warning("⚠️  No API keys available for SEC filings analysis!")
        
        # Guard for FMP free tier which 403s on SEC filings
        self.skip_sec_fmp = os.getenv("SKIP_SEC_FMP", "false").lower() == "true"
        
        # FinBERT removed — VADER fallback used for SEC sentiment analysis
        self.finbert = None
        self.finbert_tokenizer = None
        self.finbert_model = None
        
        # Initialize form parsers
        self.form_parsers = {
            '25': Form25Parser,
            '4': Form4Parser,
            '8-K': Form8KParser,
            '10-K': Form10KParser,
            '10-Q': Form10QParser,
            '144': Form144Parser,
            'DEF 14A': FormDEF14AParser,
            'SD': FormSDParser,
            '11-K': Form11KParser
        }
        
        # Form type weights for confidence calculation
        self.form_weights = {
            '10-K': 1.0,    # Annual reports
            '10-Q': 0.8,    # Quarterly reports
            '8-K': 0.6,     # Current reports
            'DEF 14A': 0.7, # Proxy statements
            '4': 0.5,       # Insider trading
            '144': 0.4,     # Planned sales
            'SD': 0.3,      # Conflict minerals
            '11-K': 0.3     # Employee stock plans
        }

        # Form type normalization map
        self.form_alias_map = {
            "10K": "10-K",
            "10Q": "10-Q",
            "8K": "8-K",
            "DEF14A": "DEF 14A",
            "11K": "11-K",
            "FORM 10-K": "10-K",
            "FORM 10-Q": "10-Q",
            "FORM 8-K": "8-K",
            "FORM DEF 14A": "DEF 14A",
            "FORM 11-K": "11-K",
            "FORM 144": "144",
            "FORM SD": "SD"
        }

    def _normalize_form_type(self, form_type: str) -> str:
        """
        Normalize form type to standard format.
        
        Args:
            form_type: Raw form type string
            
        Returns:
            Normalized form type string
        """
        if not form_type:
            return ""
            
        # Convert to uppercase and remove "FORM " prefix
        form_type = form_type.upper().replace("FORM ", "").strip()
        
        # Handle special cases
        return self.form_alias_map.get(form_type, form_type)

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using FinBERT with VADER fallback.
        
        Args:
            text: Text to analyze
            
        Returns:
            Compound sentiment score between -1 and 1
        """
        if not text or len(text.strip()) < 10:
            logger.debug("Text too short for sentiment analysis")
            return 0.0
            
        try:
            # Clean text for analysis
            text = text.strip()
            
            # If text is too long, take key sections
            if len(text) > 2048:
                # Try to extract key sections (business description, risks, etc.)
                sections = text.split('\n\n')
                important_sections = []
                
                for section in sections:
                    if any(keyword in section.lower() for keyword in [
                        'business', 'operations', 'revenue', 'profit', 'loss', 'risk',
                        'competition', 'market', 'outlook', 'guidance', 'forward', 'results'
                    ]):
                        important_sections.append(section[:500])  # Limit section size
                
                if important_sections:
                    text = ' '.join(important_sections)[:2048]
                else:
                    # Take first and last parts if no keywords found
                    text = (text[:1024] + ' ' + text[-1024:])[:2048]
            
            # VADER sentiment analysis (primary method — FinBERT removed)
            try:
                analyzer = SentimentIntensityAnalyzer()
                vader_scores = analyzer.polarity_scores(text)
                score = vader_scores['compound']
                logger.debug(f"VADER sentiment: {score:.3f}")
                return float(score)
                
            except Exception as e:
                logger.debug(f"VADER analysis failed: {e}")
            
            # Method 3: Keyword-based fallback
            positive_words = [
                'growth', 'increase', 'profit', 'revenue', 'strong', 'positive', 
                'successful', 'improvement', 'gain', 'benefit', 'opportunity',
                'optimistic', 'expansion', 'exceeded', 'outperform', 'robust'
            ]
            negative_words = [
                'loss', 'decline', 'decrease', 'risk', 'challenge', 'uncertainty',
                'negative', 'concern', 'problem', 'difficult', 'weak', 'poor',
                'disappointing', 'below', 'underperform', 'volatility'
            ]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > 0 or neg_count > 0:
                total_words = len(text.split())
                score = (pos_count - neg_count) / max(total_words / 100, 1)  # Normalize by text length
                score = max(-1, min(1, score))  # Clamp to [-1, 1]
                logger.debug(f"Keyword sentiment: {score:.3f} (pos:{pos_count}, neg:{neg_count})")
                return float(score)
            
            logger.debug("No sentiment indicators found, returning 0")
            return 0.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed completely: {e}")
            return 0.0

    def _store_filings_in_mongodb(self, ticker: str, filings_data: Dict):
        """
        Store SEC filings data in MongoDB.
        
        Args:
            ticker: Stock ticker symbol
            filings_data: Dictionary containing filings data
        """
        try:
            if not self.mongo_client:
                return
                
            # Store in sec_filings collection
            collection = self.mongo_client.db['sec_filings']
            
            # Process and store individual filings with their text and sentiment
            processed_filings = []
            for category, filings in filings_data["categorized_filings"].items():
                for filing in filings:
                    if filing.get("html_url"):
                        try:
                            # Get filing content
                            response = requests.get(filing["html_url"], timeout=30)
                            response.raise_for_status()
                            
                            # Extract text content
                            text_content = self._extract_text_from_html(response.text)
                            
                            # Analyze sentiment
                            sentiment_score = self._analyze_sentiment(text_content)
                            
                            # Store filing with its text and sentiment
                            processed_filing = {
                                'form_type': filing['Form'],
                                'filing_date': filing.get('Date', ''),
                                'text_content': text_content,
                                'sentiment_score': sentiment_score,
                                'html_url': filing['html_url'],
                                'acc': filing.get('acc', '')
                            }
                            processed_filings.append(processed_filing)
                            
                        except Exception as e:
                            logger.error(f"Error processing filing {filing.get('acc', '')}: {str(e)}")
                            continue
            
            # Add processed filings to the data
            filings_data['processed_filings'] = processed_filings
            
            # Store the complete data
            collection.replace_one(
                {'ticker': ticker, 'fetched_at': datetime.utcnow()},
                {**filings_data, 'ticker': ticker, 'fetched_at': datetime.utcnow()},
                upsert=True
            )
            logger.info(f"Stored SEC filings for {ticker} in MongoDB with {len(processed_filings)} processed filings")
            
        except Exception as e:
            logger.error(f"Error storing SEC filings in MongoDB: {str(e)}")

    @sleep_and_retry
    @limits(calls=30, period=60)
    async def fetch_kaleidoscope_filings(self, ticker: str, lookback_days: int = 30) -> Dict:
        """
        Fetch SEC filings from Kaleidoscope API.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing filings data
        """
        try:
            if not self.kaleidoscope_api_key:
                if self.skip_sec_fmp:
                    logger.info("Kaleidoscope key missing and FMP SEC skip enabled - skipping SEC filings")
                    return {"status": "skipped", "source": "kaleidoscope", "reason": "no_key"}
                logger.warning("Kaleidoscope API key not found, falling back to FMP")
                return self.fetch_fmp_filings(ticker, lookback_days)
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Construct API URL with new format
            url = f"https://api.kscope.io/v2/sec/search/{ticker}?key={self.kaleidoscope_api_key}&content=sec&sd={start_timestamp}&ed={end_timestamp}&limit=30"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 404:
                        logger.warning(f"No SEC filings found for {ticker}")
                        return {
                            "status": "error",
                            "source": "kaleidoscope",
                            "error": "No filings found"
                        }
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get('data'):
                        logger.warning(f"No SEC filings data found for {ticker}")
                        return {
                            "status": "error",
                            "source": "kaleidoscope",
                            "error": "No filings data found"
                        }
                    
                    filings = data['data']
                    total_filings = len(filings)
                    
                    if total_filings == 0:
                        return {
                            "status": "error",
                            "source": "kaleidoscope",
                            "error": "No filings found"
                        }
                    
                    # Process each filing
                    sentiments = []
                    for filing in filings:
                        # Skip non-relevant forms
                        if filing['Form'] in ['UPLOAD', 'CORRESP']:
                            continue
                            
                        # Get filing content from HTML URL
                        html_url = filing['html']
                        if not html_url:
                            continue
                            
                        try:
                            async with session.get(html_url) as html_response:
                                if html_response.status == 200:
                                    html_content = await html_response.text()
                                    # Extract text content from HTML
                                    text_content = self._extract_text_from_html(html_content)
                                    if text_content:
                                        # Analyze sentiment
                                        sentiment = self._analyze_sentiment(text_content)
                                        sentiments.append(sentiment)
                        except Exception as e:
                            logger.warning(f"Error processing filing {filing['acc']}: {str(e)}")
                            continue
                    
                    if not sentiments:
                        return {
                            "status": "error",
                            "source": "kaleidoscope",
                            "error": "No valid sentiments found"
                        }
                    
                    # Calculate aggregate sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
                    
                    # Improved confidence calculation
                    confidence = self._calculate_filing_confidence(
                        analyzed_count=len(sentiments),
                        total_filings=total_filings,
                        sentiment_std=sentiment_std,
                        avg_sentiment=abs(avg_sentiment),
                        processed_filings=processed_filings
                    )
            
                    result = {
                        "status": "success",
                        "source": "kaleidoscope",
                        "total_filings": total_filings,
                        "categorized_filings": {
                            "10-K": [f for f in filings if f['Form'] == '10-K'],
                            "10-Q": [f for f in filings if f['Form'] == '10-Q'],
                            "8-K": [f for f in filings if f['Form'] == '8-K'],
                            "DEF 14A": [f for f in filings if f['Form'] == 'DEF 14A'],
                            "4": [f for f in filings if f['Form'] == '4'],
                            "144": [f for f in filings if f['Form'] == '144'],
                            "SD": [f for f in filings if f['Form'] == 'SD'],
                            "11-K": [f for f in filings if f['Form'] == '11-K']
                        },
                        "sec_filings_sentiment": round(avg_sentiment, 4),
                        "sec_filings_volume": total_filings,
                        "sec_filings_confidence": round(confidence, 4),
                        "sec_filings_sentiment_std": round(sentiment_std, 4)
                    }
            
                    # Store in MongoDB
                    self._store_filings_in_mongodb(ticker, result)
            
                    return result
            
        except Exception as e:
            logger.error(f"Error fetching Kaleidoscope filings: {str(e)}")
            return self.fetch_fmp_filings(ticker, lookback_days)
            
    def fetch_fmp_filings(self, ticker: str, lookback_days: int = 30) -> Dict:
        """
        Fetch SEC filings from FMP API as backup.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing filings data
        """
        if self.skip_sec_fmp:
            return {"status": "skipped", "source": "fmp", "reason": "skip_env_set"}

        try:
            if not self.fmp_api_key:
                return {
                    "status": "error",
                    "source": "fmp",
                    "error": "FMP API key not found"
                }
                
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Prepare request parameters
            params = {
                "apikey": self.fmp_api_key,
                "symbol": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "limit": self.max_filings  # Limit to 30 filings
            }
            
            # Make API request
            response = requests.get("https://financialmodelingprep.com/api/v3/sec_filings", 
                                params=params, timeout=30)
            response.raise_for_status()
            filings = response.json()
            
            if not filings:
                return {
                    "status": "error",
                    "source": "fmp",
                    "error": "No filings found"
                }
                
            # Filter out insider transactions and non-relevant forms
            excluded_forms = ['4', '144', 'UPLOAD', 'CORRESP', 'NT 10-Q', 'NT 10-K', 'SC 13D/A', 'SC 13G/A']
            filings = [f for f in filings if f.get('type', '') not in excluded_forms]
            logger.info(f"FMP returned {len(filings)} total filings, processing {len(filings)} after filtering")
            total_filings = len(filings)
            
            if total_filings == 0:
                return {
                    "status": "error",
                    "source": "fmp",
                    "error": "No relevant filings found"
                }
            
            # Process and categorize filings
            categorized_filings = self._categorize_filings(filings)
            
            result = {
                "status": "success",
                "source": "fmp",
                "total_filings": total_filings,
                "categorized_filings": categorized_filings
            }
            
            # Store in MongoDB
            self._store_filings_in_mongodb(ticker, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching FMP filings: {str(e)}")
            return {
                "status": "error",
                "source": "fmp",
                "error": str(e)
            }

    def _parse_filing_content(self, html_url: str, form_type: str) -> Dict:
        """
        Parse structured data from SEC filing HTML and analyze sentiment.
        
        Args:
            html_url: URL to the HTML version of the filing
            form_type: Type of SEC form
            
        Returns:
            Dictionary containing parsed form data and sentiment
        """
        try:
            response = requests.get(html_url, timeout=30)
            response.raise_for_status()
            
            # Normalize form type
            form_type = self._normalize_form_type(form_type)
            
            # Get appropriate parser for form type
            parser_class = self.form_parsers.get(form_type)
            if parser_class:
                parser = parser_class(response.text)
                parsed_data = parser.parse()
                
                # Get text for sentiment analysis
                sentiment_text = parser.get_sentiment_text()
                if sentiment_text:
                    parsed_data['sentiment_score'] = self._analyze_sentiment(sentiment_text)
                else:
                    parsed_data['sentiment_score'] = 0.0
                    
                return parsed_data
            else:
                logger.warning(f"No parser available for form type: {form_type}")
                return {'form_type': form_type, 'error': 'No parser available'}
                
        except Exception as e:
            logger.error(f"Error parsing filing content: {str(e)}")
            return {'form_type': form_type, 'error': str(e)}

    async def analyze_filings_sentiment(self, ticker: str, lookback_days: int = 30) -> Dict:
        """
        Analyze sentiment of recent SEC filings for a ticker.
        Returns a dictionary with sentiment scores and metadata.
        Excludes insider transactions (Form 4 and Form 144) as they are handled by Finnhub.
        """
        try:
            # Check if Kaleidoscope API key is available
            if not self.kaleidoscope_api_key:
                logger.warning("Kaleidoscope API key not found, attempting FMP fallback for SEC filings")
                return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
            
            # Calculate date range - extend lookback for more filings
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=max(lookback_days, 90))  # Minimum 90 days
            
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Construct API URL with new format
            url = f"https://api.kscope.io/v2/sec/search/{ticker}?key={self.kaleidoscope_api_key}&content=sec&sd={start_timestamp}&ed={end_timestamp}&limit=50"  # Increased limit
            
            logger.info(f"Fetching SEC filings for {ticker} from Kaleidoscope API (extended {max(lookback_days, 90)} days)...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 404:
                        logger.warning(f"No SEC filings found for {ticker} via Kaleidoscope")
                        return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
                    
                    if response.status != 200:
                        logger.error(f"Kaleidoscope API returned status {response.status} for {ticker}")
                        return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
                    
                    response.raise_for_status()
                    data = await response.json()
                    logger.info(f"Kaleidoscope API response received for {ticker}")
                    
                    if not data.get('data'):
                        logger.warning(f"No SEC filings data found for {ticker} via Kaleidoscope, trying FMP fallback")
                        return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
                    
                    filings = data['data']
                    # Filter out only insider transactions and administrative forms
                    # Keep more forms for better sentiment analysis
                    excluded_forms = ['4', '144', 'UPLOAD', 'CORRESP', 'NT 10-Q', 'NT 10-K', 'SC 13D/A', 'SC 13G/A']
                    filings = [f for f in filings if f['Form'] not in excluded_forms]
                    total_filings = len(filings)
                    
                    logger.info(f"Kaleidoscope returned {len(data['data'])} total filings, processing {total_filings} after filtering")
                    
                    # Log what forms we're processing
                    form_counts = {}
                    for f in filings:
                        form_type = f['Form']
                        form_counts[form_type] = form_counts.get(form_type, 0) + 1
                    logger.info(f"Processing SEC forms: {form_counts}")
                    
                    # If still very limited filings, try FMP fallback
                    if total_filings <= 2:
                        logger.warning(f"Very few SEC filings found for {ticker} via Kaleidoscope ({total_filings}), trying FMP fallback")
                        fmp_result = await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
                        
                        # If FMP has more filings, use it; otherwise continue with Kaleidoscope
                        if fmp_result.get('sec_filings_volume', 0) > total_filings:
                            logger.info(f"FMP has more filings ({fmp_result.get('sec_filings_volume', 0)}), using FMP instead")
                            return fmp_result
                    
                    if total_filings == 0:
                        logger.warning(f"No relevant SEC filings found for {ticker} via Kaleidoscope, trying FMP fallback")
                        return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
                    
                    # Process each filing
                    sentiments = []
                    processed_filings = []
                    logger.info(f"Processing {total_filings} SEC filings for {ticker}...")
                    
                    for i, filing in enumerate(filings):
                        # Get filing content from HTML URL
                        html_url = filing['html']
                        form_type = filing.get('Form', 'Unknown')
                        filing_date = filing.get('Date', 'Unknown')
                        company_name = filing.get('Company Name', 'Unknown')
                        
                        logger.info(f"Filing {i+1}/{total_filings}: {form_type} by {company_name} on {filing_date}")
                        
                        if not html_url:
                            logger.warning(f"No HTML URL for filing {i+1}/{total_filings}: {form_type}")
                            continue
                            
                        try:
                            async with session.get(html_url) as html_response:
                                if html_response.status == 200:
                                    html_content = await html_response.text()
                                    # Extract text content from HTML
                                    text_content = self._extract_text_from_html(html_content)
                                    if text_content:
                                        # Analyze sentiment
                                        sentiment = self._analyze_sentiment(text_content)
                                        sentiments.append(sentiment)
                                        
                                        # Store processed filing for MongoDB
                                        processed_filing = {
                                            'form_type': form_type,
                                            'filing_date': filing_date,
                                            'company_name': company_name,
                                            'html_url': html_url,
                                            'html_content': html_content,
                                            'text_content': text_content,
                                            'sentiment_score': sentiment,
                                            'acc': filing.get('acc', ''),
                                            'cik': filing.get('CIK', ''),
                                            'filer': filing.get('Filer', ''),
                                            'form_desc': filing.get('Form_Desc', ''),
                                            'pdf_url': filing.get('pdf', ''),
                                            'word_url': filing.get('word', ''),
                                            'processed_at': datetime.utcnow().isoformat()
                                        }
                                        processed_filings.append(processed_filing)
                                        
                                        logger.info(f"Filing {i+1}/{total_filings}: {form_type} ({filing_date}) → sentiment: {sentiment:.3f}")
                                    else:
                                        logger.warning(f"No text content extracted from filing {i+1}/{total_filings}: {form_type}")
                                else:
                                    logger.warning(f"Failed to fetch HTML for filing {i+1}/{total_filings}: {form_type}, status: {html_response.status}")
                        except Exception as e:
                            logger.warning(f"Error processing filing {i+1}/{total_filings} ({form_type}): {str(e)}")
                            continue
                    
                    if not sentiments:
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": total_filings,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0
                        }
                    
                    # Calculate aggregate sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
                    
                    # Improved confidence calculation
                    confidence = self._calculate_filing_confidence(
                        analyzed_count=len(sentiments),
                        total_filings=total_filings,
                        sentiment_std=sentiment_std,
                        avg_sentiment=abs(avg_sentiment),
                        processed_filings=processed_filings
                    )
                    
                    logger.info(f"✓ Kaleidoscope SEC sentiment analysis complete for {ticker}: {avg_sentiment:.3f} (volume: {total_filings}, analyzed: {len(sentiments)}, confidence: {confidence:.2f})")
                    
                    # Store raw data in MongoDB if client is available
                    if self.mongo_client and processed_filings:
                        try:
                            collection = self.mongo_client.db['sec_filings_raw']
                            for filing in processed_filings:
                                collection.replace_one(
                                    {
                                        'ticker': ticker,
                                        'acc': filing['acc'],
                                        'form_type': filing['form_type'],
                                        'filing_date': filing['filing_date']
                                    },
                                    {**filing, 'ticker': ticker},
                                    upsert=True
                                )
                            logger.info(f"Stored {len(processed_filings)} SEC filings raw data for {ticker} in MongoDB")
                        except Exception as e:
                            logger.error(f"Error storing SEC raw data in MongoDB: {str(e)}")
                    
                    return {
                        "sec_filings_sentiment": round(avg_sentiment, 4),
                        "sec_filings_volume": total_filings,
                        "sec_filings_confidence": round(confidence, 4),
                        "sec_filings_analyzed": len(sentiments),
                        "sec_filings_sentiment_std": round(sentiment_std, 4),
                        "sec_filings_source": "kaleidoscope",
                        "sec_raw_data": {
                            "categorized_filings": {
                                "10-K": [f for f in filings if f['Form'] == '10-K'],
                                "10-Q": [f for f in filings if f['Form'] == '10-Q'],
                                "8-K": [f for f in filings if f['Form'] == '8-K'],
                                "25-NSE": [f for f in filings if f['Form'] == '25-NSE'],
                                "424B2": [f for f in filings if f['Form'] == '424B2'],
                                "FWP": [f for f in filings if f['Form'] == 'FWP'],
                                "DEF 14A": [f for f in filings if f['Form'] == 'DEF 14A'],
                                "SD": [f for f in filings if f['Form'] == 'SD'],
                                "11-K": [f for f in filings if f['Form'] == '11-K']
                            },
                            "total_processed": len(processed_filings),
                            "processed_at": datetime.utcnow().isoformat()
                        },
                        "sec_processed_filings": processed_filings
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing SEC filings for {ticker} via Kaleidoscope: {str(e)}, trying FMP fallback")
            return await self._analyze_fmp_filings_sentiment(ticker, lookback_days)
            
    def _is_recent(self, date_str: str) -> bool:
        """Check if a filing date is within the last 30 days."""
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            return (datetime.utcnow() - filing_date).days <= 30
        except Exception:
            return False 

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract narrative business content from SEC filing HTML for sentiment analysis.
        Focuses on MD&A, Risk Factors, Business Overview, and other narrative sections.
        
        Args:
            html_content (str): Raw HTML content from SEC filing
            
        Returns:
            str: Clean narrative content for sentiment analysis
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, and navigation elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            extracted_content = []
            
            # 1. Extract narrative sections by heading (PRIMARY for sentiment)
            narrative_sections = self._extract_narrative_sections_by_heading(soup)
            if narrative_sections:
                extracted_content.extend(narrative_sections)
                logger.info(f"Extracted {len(narrative_sections)} narrative sections: {sum(len(s) for s in narrative_sections)} characters")
            
            # 2. Extract general business narrative content (fallback)
            general_narrative = self._extract_general_narrative_content(soup)
            if general_narrative and not narrative_sections:  # Only use if no specific sections found
                extracted_content.append(general_narrative)
                logger.info(f"Extracted general narrative content: {len(general_narrative)} characters")
            
            # 3. Extract XBRL metadata for context (minimal)
            metadata_content = self._extract_minimal_metadata(soup)
            if metadata_content:
                extracted_content.append(metadata_content)
                logger.info(f"Extracted metadata context: {len(metadata_content)} characters")
            
            # Combine all content
            if extracted_content:
                combined_text = ' '.join(extracted_content)
                
                # Limit total content for sentiment analysis
                if len(combined_text) > 10000:
                    combined_text = combined_text[:10000] + "..."
                
                logger.info(f"Total extracted narrative content: {len(combined_text)} characters from SEC filing")
                return combined_text
            else:
                logger.warning("No meaningful narrative content found in SEC filing")
                return ""
            
        except Exception as e:
            logger.warning(f"Error extracting narrative text from SEC HTML: {e}")
            return ""
    
    def _extract_narrative_sections_by_heading(self, soup: BeautifulSoup) -> list:
        """
        Extract narrative sections by finding section headings and extracting content.
        Primary method for sentiment analysis content.
        """
        narrative_sections = []
        
        # Target section headings for sentiment analysis
        target_sections = [
            "Management's Discussion and Analysis",
            "MD&A",
            "Risk Factors", 
            "Business Overview",
            "Business Description",
            "Legal Proceedings",
            "Forward-Looking Statements",
            "Results of Operations",
            "Financial Condition",
            "Liquidity and Capital Resources",
            "Critical Accounting Policies",
            "Market Risk",
            "Competition",
            "Item 1A",  # Risk Factors section number
            "Item 7",   # MD&A section number  
            "Item 1"    # Business section number
        ]
        
        # Find headings and extract following content
        for heading_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'div', 'p', 'span', 'font', 'b', 'strong']):
            heading_text = heading_tag.get_text(strip=True)
            
            # Check if this heading matches our target sections
            for target in target_sections:
                if (target.lower() in heading_text.lower() and 
                    len(heading_text) < 300 and  # Reasonable heading length
                    len(heading_text) > 5):      # Not too short
                    
                    # Extract content following this heading
                    section_content = self._extract_content_after_heading(heading_tag)
                    if section_content and len(section_content) > 500:  # Substantial content only
                        narrative_sections.append(section_content)
                        logger.info(f"Found section: {target} ({len(section_content)} chars)")
                    break
        
        return narrative_sections
    
    def _extract_content_after_heading(self, heading_element) -> str:
        """Extract text content following a section heading until next heading of same level."""
        content_parts = []
        current = heading_element.next_sibling
        heading_level = self._get_heading_level(heading_element)
        
        while current and len(' '.join(content_parts)) < 3000:  # Limit per section
            if hasattr(current, 'name'):
                # Stop if we hit another heading of same or higher level
                if current.name in ['h1', 'h2', 'h3', 'h4']:
                    current_level = self._get_heading_level(current)
                    if current_level <= heading_level:
                        break
                
                # Extract text from paragraphs and divs
                if current.name in ['p', 'div', 'span', 'font']:
                    text = current.get_text(strip=True)
                    if (len(text) > 100 and 
                        not self._is_boilerplate_text(text) and
                        self._contains_business_keywords(text)):
                        content_parts.append(text)
            
            current = current.next_sibling
        
        return ' '.join(content_parts)
    
    def _get_heading_level(self, element) -> int:
        """Get numeric level of heading element."""
        if element.name == 'h1': return 1
        elif element.name == 'h2': return 2  
        elif element.name == 'h3': return 3
        elif element.name == 'h4': return 4
        else: return 5  # div, p, span treated as lower level
    
    def _extract_general_narrative_content(self, soup: BeautifulSoup) -> str:
        """Extract general narrative content as fallback when specific sections not found."""
        narrative_content = []
        
        # Look for substantial text blocks (paragraphs, divs with significant content)
        for element in soup.find_all(['p', 'div']):
            text = element.get_text(strip=True)
            
            # Filter for substantial business-relevant content
            if (len(text) > 300 and 
                not self._is_boilerplate_text(text) and
                self._contains_business_keywords(text)):
                narrative_content.append(text)
                
                # Limit total fallback content
                if len(' '.join(narrative_content)) > 5000:
                    break
        
        return ' '.join(narrative_content)
    
    def _extract_minimal_metadata(self, soup: BeautifulSoup) -> str:
        """Extract minimal company metadata for context."""
        metadata_parts = []
        
        # Extract company name from XBRL
        for element in soup.find_all(['ix:nonnumeric']):
            name = element.get('name', '')
            if 'EntityRegistrantName' in name or 'EntityName' in name:
                text = element.get_text(strip=True)
                if text and len(text) > 5:
                    metadata_parts.append(f"Company: {text}")
                    break  # Only need one company name
        
        return ' '.join(metadata_parts)
    
    def _is_boilerplate_text(self, text: str) -> bool:
        """Check if text is SEC boilerplate that should be filtered out."""
        boilerplate_phrases = [
            'securities and exchange commission',
            'washington, d.c.', 'united states',
            'commission file number', 'exact name of registrant',
            'pursuant to section', 'table of contents',
            'securities act of 1933', 'exchange act of 1934',
            'incorporated by reference', 'form 10-k',
            'form 10-q', 'form 8-k', 'proxy statement',
            'annual report', 'quarterly report'
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in boilerplate_phrases)
    
    def _contains_business_keywords(self, text: str) -> bool:
        """Check if text contains relevant business/sentiment keywords."""
        business_keywords = [
            'revenue', 'income', 'profit', 'loss', 'earnings',
            'growth', 'decline', 'increase', 'decrease',
            'performance', 'results', 'operations', 'business',
            'market', 'competition', 'strategy', 'outlook',
            'risk', 'opportunity', 'challenge', 'financial',
            'customers', 'sales', 'products', 'services',
            'management', 'believe', 'expect', 'anticipate',
            'future', 'prospects', 'trends', 'industry'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in business_keywords)

    async def _analyze_fmp_filings_sentiment(self, ticker: str, lookback_days: int = 30) -> Dict:
        """
        Fallback method to analyze SEC filings sentiment using FMP API.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            if self.skip_sec_fmp:
                logger.info(f"FMP SEC fallback disabled via SKIP_SEC_FMP for {ticker}")
                return {
                    "sec_filings_sentiment": 0.0,
                    "sec_filings_volume": 0,
                    "sec_filings_confidence": 0.0,
                    "sec_filings_analyzed": 0,
                    "sec_filings_error": "Skipped per environment config"
                }

            if not self.fmp_api_key:
                logger.warning("FMP API key not found, SEC filings analysis unavailable")
                return {
                    "sec_filings_sentiment": 0.0,
                    "sec_filings_volume": 0,
                    "sec_filings_confidence": 0.0,
                    "sec_filings_analyzed": 0,
                    "sec_filings_error": "No API keys available for SEC filings"
                }
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # FMP SEC filings endpoint
            url = f"https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&limit=30&apikey={self.fmp_api_key}"
            
            logger.info(f"Fetching SEC filings for {ticker} from FMP API...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"FMP API returned status {response.status} for {ticker} SEC filings")
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0,
                            "sec_filings_error": f"FMP API error: {response.status}"
                        }
                    
                    data = await response.json()
                    logger.info(f"FMP API response received for {ticker}: {len(data) if data else 0} filings")
                    
                    if not data:
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0,
                            "sec_filings_error": "No filings found via FMP"
                        }
                    
                    # Filter out insider transactions and non-relevant forms
                    excluded_forms = ['4', '144', 'UPLOAD', 'CORRESP', 'NT 10-Q', 'NT 10-K', 'SC 13D/A', 'SC 13G/A']
                    filings = [f for f in data if f.get('type', '') not in excluded_forms]
                    logger.info(f"FMP returned {len(data)} total filings, processing {len(filings)} after filtering")
                    total_filings = len(filings)
                    
                    if total_filings == 0:
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0,
                            "sec_filings_error": "No relevant filings found"
                        }
                    
                    # For FMP, we might not get HTML URLs, so we'll do basic sentiment based on form types
                    # More positive sentiment for positive form types, neutral for others
                    sentiments = []
                    for filing in filings[:10]:  # Limit to recent 10 filings
                        form_type = filing.get('type', '')
                        filing_date = filing.get('fillingDate', '')
                        
                        # Basic sentiment based on form type (this is a simplified approach)
                        if form_type in ['10-K', '10-Q']:
                            # Regular reports - neutral to slightly positive
                            sentiment = 0.1
                        elif form_type == '8-K':
                            # Current reports - could be positive or negative, default neutral
                            sentiment = 0.0
                        elif form_type == 'DEF 14A':
                            # Proxy statements - slightly positive (good governance)
                            sentiment = 0.05
                        else:
                            sentiment = 0.0
                        
                        sentiments.append(sentiment)
                        logger.info(f"FMP SEC filing: {form_type} dated {filing_date}, sentiment: {sentiment}")
                    
                    if not sentiments:
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": total_filings,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0,
                            "sec_filings_error": "No sentiments calculated"
                        }
                    
                    # Calculate aggregate sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Lower confidence for FMP since we're not doing full text analysis
                    confidence = min(len(sentiments) / 10, 0.5)  # Max 0.5 confidence for FMP
                    
                    logger.info(f"FMP SEC sentiment analysis complete for {ticker}: {avg_sentiment:.3f} (confidence: {confidence:.2f})")
                    
                    # Store raw data in MongoDB if client is available
                    if self.mongo_client and filings:
                        try:
                            collection = self.mongo_client.db['sec_filings_raw']
                            for filing in filings[:10]:  # Store the processed filings
                                processed_filing = {
                                    'form_type': filing.get('type', ''),
                                    'filing_date': filing.get('fillingDate', ''),
                                    'company_name': filing.get('companyName', ''),
                                    'link': filing.get('link', ''),
                                    'final_link': filing.get('finalLink', ''),
                                    'cik': filing.get('cik', ''),
                                    'accepted_date': filing.get('acceptedDate', ''),
                                    'period_of_report': filing.get('periodOfReport', ''),
                                    'effective_date': filing.get('effectiveDate', ''),
                                    'processed_at': datetime.utcnow().isoformat(),
                                    'source': 'fmp'
                                }
                                collection.replace_one(
                                    {
                                        'ticker': ticker,
                                        'cik': filing.get('cik', ''),
                                        'form_type': filing.get('type', ''),
                                        'filing_date': filing.get('fillingDate', '')
                                    },
                                    {**processed_filing, 'ticker': ticker},
                                    upsert=True
                                )
                            logger.info(f"Stored {len(filings[:10])} FMP SEC filings raw data for {ticker} in MongoDB")
                        except Exception as e:
                            logger.error(f"Error storing FMP SEC raw data in MongoDB: {str(e)}")
                    
                    return {
                        "sec_filings_sentiment": round(avg_sentiment, 4),
                        "sec_filings_volume": total_filings,
                        "sec_filings_confidence": round(confidence, 4),
                        "sec_filings_analyzed": len(sentiments),
                        "sec_filings_source": "fmp",
                        "sec_raw_data": {
                            "categorized_filings": {
                                "10-K": [f for f in filings if f.get('type') == '10-K'],
                                "10-Q": [f for f in filings if f.get('type') == '10-Q'],
                                "8-K": [f for f in filings if f.get('type') == '8-K'],
                                "DEF 14A": [f for f in filings if f.get('type') == 'DEF 14A'],
                                "other": [f for f in filings if f.get('type') not in ['10-K', '10-Q', '8-K', 'DEF 14A']]
                            },
                            "total_processed": len(sentiments),
                            "processed_at": datetime.utcnow().isoformat()
                        }
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing FMP SEC filings for {ticker}: {str(e)}")
            return {
                "sec_filings_sentiment": 0.0,
                "sec_filings_volume": 0,
                "sec_filings_confidence": 0.0,
                "sec_filings_analyzed": 0,
                "sec_filings_error": str(e)
            } 

    def _calculate_filing_confidence(self, analyzed_count: int, total_filings: int, 
                                   sentiment_std: float, avg_sentiment: float, 
                                   processed_filings: List[Dict]) -> float:
        """
        Calculate confidence score for SEC filing sentiment analysis.
        
        Args:
            analyzed_count: Number of filings successfully analyzed
            total_filings: Total number of filings found
            sentiment_std: Standard deviation of sentiment scores
            avg_sentiment: Average absolute sentiment score
            processed_filings: List of processed filing data
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Volume factor: More analyzed filings = higher confidence
            volume_factor = min(analyzed_count / 8, 1.0)  # Optimal at 8+ filings
            
            # Coverage factor: How many of available filings were analyzed
            coverage_factor = analyzed_count / max(total_filings, 1) if total_filings > 0 else 0
            
            # Consistency factor: Lower standard deviation = higher confidence  
            consistency_factor = max(0, 1 - (sentiment_std / 0.5))  # Normalize by 0.5 std
            
            # Content quality factor: Based on filing types and text length
            quality_scores = []
            for filing in processed_filings:
                form_type = filing.get('form_type', '')
                text_content = filing.get('text_content', '')
                
                # Assign quality scores based on filing type
                type_scores = {
                    '10-K': 1.0,  # Annual reports - highest quality
                    '10-Q': 0.9,  # Quarterly reports - high quality
                    '8-K': 0.7,   # Current reports - medium quality
                    'DEF 14A': 0.6,  # Proxy statements - medium quality
                    'SD': 0.4,    # Specialized disclosure - lower quality
                    '4': 0.3,     # Insider trading - minimal sentiment value
                    '144': 0.2    # Intent to sell - minimal sentiment value
                }
                
                type_score = type_scores.get(form_type, 0.5)  # Default medium quality
                
                # Text length factor: More text usually = better analysis
                text_length = len(text_content) if text_content else 0
                length_score = min(text_length / 1000, 1.0)  # Normalize by 1000 chars
                
                filing_quality = (type_score + length_score) / 2
                quality_scores.append(filing_quality)
            
            quality_factor = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            
            # Signal strength factor: Stronger sentiment signals = higher confidence
            signal_strength = min(avg_sentiment / 0.3, 1.0)  # Normalize by 0.3 sentiment
            
            # Weighted confidence calculation
            confidence = (
                volume_factor * 0.25 +      # 25% weight on volume
                coverage_factor * 0.20 +    # 20% weight on coverage  
                consistency_factor * 0.25 + # 25% weight on consistency
                quality_factor * 0.20 +     # 20% weight on content quality
                signal_strength * 0.10      # 10% weight on signal strength
            )
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            logger.debug(f"SEC confidence factors: volume={volume_factor:.2f}, coverage={coverage_factor:.2f}, "
                        f"consistency={consistency_factor:.2f}, quality={quality_factor:.2f}, "
                        f"signal={signal_strength:.2f}, final={confidence:.2f}")
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating filing confidence: {e}")
            # Fallback to simple confidence calculation
            volume_factor = min(analyzed_count / 5, 1.0)
            consistency_factor = max(0, 1 - sentiment_std) if sentiment_std else 0.5
            return (volume_factor + consistency_factor) / 2 