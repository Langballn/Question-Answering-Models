import utils
import argparse
from utils.setup import download_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--dir', required=True)
    opt = parser.parse_args()

    download_embeddings(opt.url, opt.filename, opt.dir)
