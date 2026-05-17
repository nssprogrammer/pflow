"""PFlow-T dataset version: persistence-based forward process on real MNIST.

Forward process: kill H1 features (holes) of an MNIST digit in ascending
order of persistence, by filling them in with intensity at the death-cell
location.

Inference: x_0-parameterization with one-shot sampling at evaluation time.
"""

__version__ = "0.2.0"
