import galois
import numpy as np

bch = galois.BCH(n=511, k=493)


def encode(data):
    return bch.encode(data)


def decode(received):
    return bch.decode(received)


def main() -> None:
    data = np.random.randint(0, 2, size=493)
    encoded = encode(data)
    decoded = decode(encoded)
    print("Original data:", data)
    print("Encoded data:", encoded)
    print("Decoded data:", decoded)
