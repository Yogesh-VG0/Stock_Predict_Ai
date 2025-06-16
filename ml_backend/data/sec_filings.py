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
from transformers import pipeline
import aiohttp
import statistics

# Load environment variables
load_dotenv()

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
        
        # Initialize FinBERT for sentiment analysis
        try:
            self.finbert = pipeline(
                "sentiment-analysis",
                model="yiyanghkust/finbert-tone",
                framework="pt",
                device=-1
            )
            logger.info("Loaded FinBERT model for sentiment analysis")
        except Exception as e:
            logger.error(f"Error initializing FinBERT: {str(e)}")
            self.finbert = None
        
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
        if not text:
            return 0.0
            
        try:
            if self.finbert:
                # Limit text length to prevent OOM
                if len(text) > 1024:
                    text = text[:1024]
                    
                # Use FinBERT for financial text
                result = self.finbert(text[:512])[0]  # FinBERT has 512 token limit
                label = result['label']
                if label == 'positive':
                    return 0.8
                elif label == 'negative':
                    return -0.8
                else:
                    return 0.0
            else:
                # Fallback to VADER
                return self.vader.polarity_scores(text)['compound']
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Fallback to VADER if FinBERT fails
            return self.vader.polarity_scores(text)['compound']

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
                    
                    # Calculate confidence based on volume and consistency
                    volume_factor = min(len(sentiments) / 10, 1.0)  # Cap at 10 filings
                    consistency_factor = 1 - min(sentiment_std, 1.0)
                    confidence = (volume_factor + consistency_factor) / 2
            
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
                
            # Process and categorize filings
            categorized_filings = self._categorize_filings(filings)
            
            result = {
                "status": "success",
                "source": "fmp",
                "total_filings": len(filings),
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
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0
                        }
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get('data'):
                        logger.warning(f"No SEC filings data found for {ticker}")
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0
                        }
                    
                    filings = data['data']
                    # Filter out insider transactions (Form 4 and Form 144)
                    filings = [f for f in filings if f['Form'] not in ['4', '144', 'UPLOAD', 'CORRESP']]
                    total_filings = len(filings)
                    
                    if total_filings == 0:
                        return {
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": 0,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0
                        }
                    
                    # Process each filing
                    sentiments = []
                    for filing in filings:
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
                            "sec_filings_sentiment": 0.0,
                            "sec_filings_volume": total_filings,
                            "sec_filings_confidence": 0.0,
                            "sec_filings_analyzed": 0
                        }
                    
                    # Calculate aggregate sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
                    
                    # Calculate confidence based on volume and consistency
                    volume_factor = min(len(sentiments) / 10, 1.0)  # Cap at 10 filings
                    consistency_factor = 1 - min(sentiment_std, 1.0)
                    confidence = (volume_factor + consistency_factor) / 2
                    
                    return {
                        "sec_filings_sentiment": round(avg_sentiment, 4),
                        "sec_filings_volume": total_filings,
                        "sec_filings_confidence": round(confidence, 4),
                        "sec_filings_analyzed": len(sentiments),
                        "sec_filings_sentiment_std": round(sentiment_std, 4)
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing SEC filings for {ticker}: {str(e)}")
            return {
                "sec_filings_sentiment": 0.0,
                "sec_filings_volume": 0,
                "sec_filings_confidence": 0.0,
                "sec_filings_analyzed": 0,
                "sec_filings_error": str(e)
            }
            
    def _is_recent(self, date_str: str) -> bool:
        """Check if a filing date is within the last 30 days."""
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            return (datetime.utcnow() - filing_date).days <= 30
        except Exception:
            return False 

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content using BeautifulSoup.
        
        Args:
            html_content (str): Raw HTML content from SEC filing
            
        Returns:
            str: Clean text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error extracting text from HTML: {e}")
            return "" 