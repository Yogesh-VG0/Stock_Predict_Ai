from setuptools import setup, find_packages

setup(
    name="stockpredict-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-jose[cryptography]",
        "redis",
        "fastapi-limiter",
        "pymongo",
        "pandas",
        "numpy",
        "scikit-learn",
        "optuna",
        "python-dotenv",
        "praw",
        "tweepy",
        "feedparser",
        "vaderSentiment",
        "transformers",
        "beautifulsoup4",
        "requests",
        "ta"
    ],
    extras_require={
        "ml": ["tensorflow>=2.0.0"]
    }
) 