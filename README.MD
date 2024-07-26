Uses the SVD approach of the original authors, given its superior performance, to dimensionality reduce the GEX data. Then uses a MLP architecture to predict protein abundance measurements from GEX data.

Example usage: python3 main.py --model FFNN --desc 'SVD-Normed RNA; MLP-Based predictions' --epochs=1