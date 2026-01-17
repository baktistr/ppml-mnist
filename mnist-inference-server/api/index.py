"""
Vercel entry point for FastAPI application
This file is required for Vercel to properly serve the FastAPI app
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the app
from main import app

# Vercel will use this as the entry point
handler = app
