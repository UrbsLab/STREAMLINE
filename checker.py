import sys
import argparse
from streamline.utils.checker import check_phase


def main(argv):
    parser = argparse.ArgumentParser(description='program to check if run is complete')
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--phase', dest='phase', type=int, default=5, help='phase to check')
    parser.add_argument('--count-only', dest='len_only', type=bool, default=False, help='show only no of jobs')
    parser.add_argument('--rep-data-path', dest='rep_data_path', type=str, default='',
                        help='replication dataset path')
    parser.add_argument('--dataset-for-rep', dest='dataset_for_rep',
                        type=str, default='', help='train dataset for replication path')

    options = parser.parse_args(argv[1:])
    check_phase(options.output_path, options.experiment_name, options.phase, options.len_only,
                options.rep_data_path, options.dataset_for_rep)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
