import sys
import argparse
from importlib.metadata import version
from contextcountbooster.encoder import OneHotEncoder
from contextcountbooster.train_boosting import Booster
from contextcountbooster.predict import Predicter


def get_parser():
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """

    parser = argparse.ArgumentParser(
        prog="ContextCountBooster",
        description="""
                    Sequence context count modeling with weights using xgboost
                    """,
        usage="ccb [-h] [-V] [COMMAND [COMMAND_OPTS...]] ",
    )

    v = version("contextcountbooster")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {v}")
    subparsers = parser.add_subparsers(
        dest="command", prog="ccb", title="Valid commands", help="Command description"
    )

    pre_proc = subparsers.add_parser(
        name="encode",
        help="Preprocess the input data: contexts are one hot encoded and pseudocounts added to counts and weights",
        description="Preprocess the input data: contexts are one hot encoded and pseudocounts added to counts and weights",
    )
    pre_proc.add_argument(
        "input_counts",
        help="Input TSV file with count data, containing sequence context and the respective count",
    )
    pre_proc.add_argument(
        "input_weights",
        help="Input TSV file with weight data, containing sequence context and the respective weight",
    )
    pre_proc.add_argument(
        "--output_prefix", help="Output file start prefix", default=""
    )
    pre_proc.add_argument(
        "--output_dir",
        help="Directory where to write the output files, if not provided, output will be written to the current directory.",
        default=None,
    )
    pre_proc.add_argument(
        "--encoding",
        help="Whether to use the 4-bit or 7-bit encoder",
        default=7,
        type=int,
        choices=[4, 7],
    )
    pre_proc.add_argument(
        "--ref_base",
        help="Unique target middle base of the input contexts. '\
                          The input will be filtered to only include context with REF in the middle. '\
                          If not provided, input should contain a single unique middle base.",
        default=None,
    )

    train = subparsers.add_parser(
        name="train",
        help="Train boosting model based on input context counts and weights",
        description="Train boosting model based on input context counts and weights",
    )

    train.add_argument(
        "--train_data",
        help="Input TSV training data (preprocessed with encode command)",
    )
    train.add_argument(
        "--dist_CV",
        help="Whether CV is being run in a distributed way from the pipeline",
        action="store_true",
    )
    train.add_argument(
        "--aggregate_and_train_only",
        help="CV done in a distributed manner, aggregate results and train model on full training set",
        action="store_true",
    )
    train.add_argument(
        "--encoding",
        help="Whether to use the 4-bit or 7-bit encoder",
        default=7,
        type=int,
        choices=[4, 7],
    )
    train.add_argument(
        "--CV_res",
        help="Save cross-validation results range (min, q25, median, q75, max)",
        action="store_true",
    )
    train.add_argument("--n_CV_it", help="Number of CV iterations", default=1, type=int)
    train.add_argument(
        "--output_dir",
        help="Directory to write the model and training statistics in. Current directory by default",
        default="./",
    )
    train.add_argument(
        "--n_CV_folds", help="Number of cross-validation folds", default=5, type=int
    )
    train.add_argument(
        "--max_depth", help="Maximum depth of a tree.", nargs="*", default="k_based"
    )  # [4, 6, 8, 10, 14, 18, 22]
    train.add_argument(
        "--eta",
        help="Learning rate: step size shrinkage used in update to prevent overfitting; range: [0, 1]",
        nargs="*",
        default=[0.1],
    )  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    train.add_argument(
        "--subsample",
        help="Subsample ratio of the training instances per boosting iteration.",
        nargs="*",
        default=[0.5],
    )
    train.add_argument(
        "--colsample_bytree",
        help="Subsample ratio of features when constructing each tree",
        nargs="*",
        default=[0.5],
    )
    train.add_argument(
        "--colsample_bylevel",
        help="Subsample ratio of features when constructing each tree level",
        nargs="*",
        default=[0.5],
    )
    train.add_argument(
        "--l2_lambda",
        help="L2 regularization term on weights. Increasing this value will make model more conservative.",
        nargs="*",
        default=[1],
    )  # [1, 2, 5, 10, 20, 30]
    train.add_argument(
        "--tree_method",
        help="The tree construction algorithm used",
        nargs="*",
        default=["auto"],
    )
    train.add_argument(
        "--grow_policy",
        help="Controls a way new nodes are added to the tree '\
                       (depthwise=split at nodes closest to the root; lossguide=split at nodes with highest loss change)",
        nargs="*",
        default=["lossguide"],
    )
    train.add_argument(
        "--alpha",
        help="Pseudocount fraction (n) with respect to the total sample size (N), alpha=n/N",
        default=0.01,
    )

    predict = subparsers.add_parser(
        name="predict",
        help="Predict frequencies using the trained model.",
        description="Predict frequencies using the trained model.",
    )
    predict.add_argument(
        "test_data", help="Input TSV test data (preprocessed with encode command)"
    )
    predict.add_argument("model", help="Trained xgb model")
    predict.add_argument(
        "null_model", help="Null model: the mean frequency of the training data."
    )
    predict.add_argument(
        "--output_dir",
        help="Directory to write the model and training statistics in. Current directory by default",
        default="./",
    )

    return parser


def main(args=None):
    parser = get_parser()
    opts = parser.parse_args(args=args)

    if opts.command is None:
        parser.print_help(sys.stderr)
        return 1

    print(f"Command line arguments: {opts}")
    if opts.command == "encode":
        encoder = OneHotEncoder(
            opts.input_counts,
            opts.input_weights,
            opts.output_dir,
            opts.output_prefix,
            opts.encoding,
            opts.ref_base,
        )
        encoder.encode()

    elif opts.command == "train":
        booster = Booster(
            opts.train_data,
            opts.output_dir,
            opts.dist_CV,
            opts.aggregate_and_train_only,
            opts.encoding,
            opts.CV_res,
            opts.n_CV_it,
            opts.n_CV_folds,
            opts.max_depth,
            opts.eta,
            opts.subsample,
            opts.colsample_bytree,
            opts.colsample_bylevel,
            opts.l2_lambda,
            opts.tree_method,
            opts.grow_policy,
            opts.alpha,
        )
        model = booster.train_booster()
        if not opts.dist_CV:
            booster.plot_feature_gain(model)
            booster.plot_feature_weight(model)

    elif opts.command == "predict":
        predicter = Predicter(
            opts.test_data, opts.model, opts.null_model, opts.output_dir
        )
        predicter.predict()

    else:
        print(f"Wrong command: {opts.command}")
        return 1

    return 0
