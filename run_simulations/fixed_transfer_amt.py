from opt_targeted_transfers import (
    UnconditionalDiscreteTransfers,
    UnconditionalTargetedTransfers,
)
from data_loaders import get_dataset
from data_utils import split_data
from reporting import write_result
import argh


@argh.arg("--nclass", nargs="+", type=int)
@argh.arg("--tolerance", default=0.1)
@argh.arg("--d", default=2)
@argh.arg("--country", default="malawi")
@argh.arg("--save", default="results")
def main(
    country="malawi",
    d=2,
    tolerance=None,
    nclass=2,
    save="malawi_nclass_results.csv",
):
    X, y, r, features = get_dataset(country)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(
        X=X[:, :d], y=y, r=None, p=0.6
    )  # for now not using sampling weights r

    tt = UnconditionalTargetedTransfers(name=country, tolerance=tolerance, c_bar=2.15)
    tt.fit(X_train, y_train, r_train)
    tt.run_opt(X_test, r_test, path="{}_d={}_tol={}.csv".format(country, d, tolerance))
    res = tt.evaluate(X_test, y_test, r_test)
    write_result(save + "{}.csv".format(country), res)

    dt = UnconditionalDiscreteTransfers(
        method="lindsey", nclass=None, c_bar=2.15, tolerance=tolerance
    )
    dt.fit(X_train, y_train, r_train)

    for n_c in nclass:
        dt.set_nclass(n_c)
        dt.run_opt(X_test, r_test)
        res = dt.evaluate(X_test, y_test, r_test)
        write_result(save + "{}.csv".format(country), res)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
