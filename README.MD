Uses the SVD approach of the original authors, given its superior performance, to dimensionality reduce the GEX data. Then uses a conditional variational autoencoder architecture to predict protein abundance measurements from GEX data.


Example usage: python3 main.py --model cvae --desc 'SVD-Normed RNA; CVAE-Based predictions' --epochs=1
