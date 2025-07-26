#!/bin/bash
echo "Updating LangChain packages to compatible versions..."
pip install -U langchain langchain-community langchain-text-splitters langchain-huggingface
echo "Package update complete!"
