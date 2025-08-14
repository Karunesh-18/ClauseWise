#!/usr/bin/env python3
"""
Setup script for Document Analyzer with LayoutLM
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("🚀 Setting up Document Analyzer with LayoutLM...")
    print("=" * 50)
    
    # Required packages
    packages = [
        "streamlit",
        "PyPDF2", 
        "python-docx",
        "pandas",
        "transformers",
        "torch",
        "pillow",
        "pytesseract",
        "pdf2image"
    ]
    
    print("📦 Installing Python packages...")
    for package in packages:
        try:
            print(f"Installing {package}...")
            install_package(package)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            continue
    
    print("\n🔍 System Requirements:")
    print("For OCR functionality, you need Tesseract OCR installed:")
    print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  macOS: brew install tesseract")  
    print("  Linux: sudo apt-get install tesseract-ocr")
    
    print("\n🎉 Setup complete!")
    print("Run the app with: streamlit run analyzer.py")

if __name__ == "__main__":
    main()
