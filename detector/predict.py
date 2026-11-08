"""Prédiction terminal : python -m detector.predict 'Votre texte...'"""

import argparse
import json

from .inference import analyze, load_detector


def main():
    parser = argparse.ArgumentParser(description="Analyse un texte français dans le terminal.")
    parser.add_argument("text", help="Texte à analyser (entre guillemets)")
    args = parser.parse_args()
    model, metadata = load_detector()
    print(json.dumps(analyze(args.text, model, metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
