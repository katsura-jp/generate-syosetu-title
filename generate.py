import os
import sys
sys.path.append("./src/")
import argparse

from generator import TitleGenerator

def main(args):
    generator = TitleGenerator(args.ckpt, args.beam_size)

    with open(args.input, 'r', encoding='UTF-8') as f:
        story = f.read()

    print(f"\n【あらすじ】\n{story}\n\n")

    ret, title = generator.generate(story)
    if ret:
        print(f"【タイトル】 {title} \n\n")
    else:
        print(f"正しく推論できませんでした．")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--input", "-i", type=str, help="path/to/input.txt")
    parser.add_argument('--beam_size', '-b', type=int, default=5,
                        help="Beam size for searching")
    args = parser.parse_args()
    main(args)