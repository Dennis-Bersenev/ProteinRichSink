Uses the SVD approach of the original authors, given its superior performance, to dimensionality reduce the GEX data. Then uses a MLP architecture to predict protein abundance measurements from GEX data.


Example usage: 
1. python3 main.py --model MLP --desc 'SVD-Normed RNA; MLP-Based predictions' --epochs=1

2. python3 main.py --model SH --desc 'SVD-Normed RNA; MLP-Sinkhorn modified predictions' --epochs=100
