from coach_pl.tool.train import arg_parser, main

import diffusion

if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)
