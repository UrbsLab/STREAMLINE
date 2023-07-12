# import argparse

# parser = argparse.ArgumentParser(description="",
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # Arguments with no defaults - Global Args
# parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
# parser.add_argument('--exp-name', dest='experiment_name', type=str,
#                     help='name of experiment output folder (no spaces)')

# args, unknown = parser.parse_known_args(argv[1:])

# print(args['out_path'])

import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--filename', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument

f = open(args.filename + ".txt", "a")
f.write('Hello,' + args.filename)
f.close()



