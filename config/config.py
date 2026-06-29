"""
Configuration file for AI-VulScanner project
Loads settings from environment variables with fallback defaults
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, will use os.environ directly
    pass

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
URLS_DATA_DIR = os.path.join(DATA_DIR, "urls")

# File paths
GROUND_TRUTH_FILE = os.path.join(RAW_DATA_DIR, "expectedresults-1.2.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "test_level_dataset.csv")
TESTS_CSV = os.path.join(URLS_DATA_DIR, "tests.csv")

# OWASP Benchmark Configuration (from environment or defaults)
BENCHMARK_PROTOCOL = os.getenv("BENCHMARK_PROTOCOL", "https")
BENCHMARK_HOST = os.getenv("BENCHMARK_HOST", "localhost")
BENCHMARK_PORT = os.getenv("BENCHMARK_PORT", "8443")
BASE_URL = f"{BENCHMARK_PROTOCOL}://{BENCHMARK_HOST}:{BENCHMARK_PORT}/benchmark/"

# ZAP Configuration (from environment or defaults)
ZAP_API_KEY = os.getenv("ZAP_API_KEY", "simukc8c9mu1ldknevn3624nb9")
ZAP_PROXY_HOST = os.getenv("ZAP_PROXY_HOST", "127.0.0.1")
ZAP_PROXY_PORT = os.getenv("ZAP_PROXY_PORT", "8081")

ZAP_PROXY = {
    "http": f"http://{ZAP_PROXY_HOST}:{ZAP_PROXY_PORT}",
    "https": f"http://{ZAP_PROXY_HOST}:{ZAP_PROXY_PORT}",
}
