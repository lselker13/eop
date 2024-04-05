import pandas as pd

PATH_TO_DATA = "/Users/leo/Documents/gpl/eop/data/malawi/malawi_merged_shruthi.csv"
CONVERSION_FACTORS = {"malawi": 0.01406191874}


def get_dataset(country_name):

    if country_name == "uganda":
        return _load_uganda_data()
    elif country_name == "ethiopia":
        return _load_ethiopia_data()
    elif country_name == "malawi":
        return _load_malawi_data()


def _load_uganda_data():
    pass


def _load_ethiopia_data():
    pass


def _load_malawi_data():
    df = pd.read_csv(PATH_TO_DATA)

    # Drop rows of dataframe that missing outcome values
    df.dropna(axis=0, subset="rexpagg", inplace=True)
    df = df.reset_index()

    # Convert outcome to consumption per capita per day in terms of 2017 USD
    #    1. Use Shruthi's conversion factor's to convert to 2017 USD
    #    2. Convert household consumption to consumption per capita (using adult equivalence scale).
    #       If adult equiv is NaN, then impute the mean.
    #    3. Convert consumption to consumption per day
    # Note that datasets may differ in whether they report yearly or monthly consumption,
    # or their adult equivalence scale.
    df["outcome"] = df["rexpagg"].copy()
    df["outcome"] *= CONVERSION_FACTORS["malawi"]
    adult_equiv_per_hh = (
        df["num_kids"] + df["num_adults"]
    )  # May prefer alpha * num_kids + num_adults for alpha in (0, 1)
    adult_equiv_per_hh = adult_equiv_per_hh.fillna(adult_equiv_per_hh.mean())
    df["outcome"] /= adult_equiv_per_hh
    df["outcome"] /= 365
    y = df["outcome"]

    # Select features from the dataset. For Malawi, we consider dwelling type, wall type,
    # number of kids in the household, number of adults in the household,
    # distance to the nearest market, distance to the nearest borderpost,
    # binary indicators for generator, electric stove, kerosene stove, radio.

    categorical_features = ["dwelling_type", "wall_type"]
    other_features = [
        "num_kids",
        "num_adults",
        "dist_admarc",
        "dist_borderpost",
        "yn_generator",
        "yn_elec_stove",
        "yn_ker_stove",
        "yn_radio",
    ]

    # Handle missingness in X's by creating dummy variables to denote missingness in the features
    # with at least 15% missingness. Otherwise, just impute with mean.
    # For categorical variables, use a one hot encoding with an
    # additional category that is a NaN indicator for that feature.
    # If there are NaNs in other feature, add a new category that is a
    # NaN for that feature and impute the NaN with 0.
    X_cat = []
    X_cat = [
        pd.get_dummies(
            df[[cat_feat]],
            dummy_na=(df[[cat_feat]].isna().sum().item() > 0.15 * len(df[cat_feat])),
        ).astype(float)
        for cat_feat in categorical_features
    ]
    for col in other_features:
        if df[col].isna().sum() > 0.15 * len(df):
            df[f"{col}_nan"] = df[col].isna().astype(float)
            other_features.append(f"{col}_nan")
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(df[col].mean())
    X_other = df[sorted(other_features)]
    X = X_cat[0].join(X_cat[1:])
    X = X.join(X_other)

    # Get survey weights
    r = df["hh_wgt"]
    return X.to_numpy(), y.to_numpy(), r.to_numpy(), X.columns
