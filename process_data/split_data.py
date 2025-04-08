import pandas as pd
import argh
from pathlib import Path

from sklearn.model_selection import train_test_split


@argh.arg("--in_path", type=str)
@argh.arg("--out_path", type=str)
@argh.arg("--train_fraction", type=float, default=0.6)
@argh.arg("--random_seed", type=int, default=1234)
@argh.arg("--stratify", type=str, default='ea_id')
def main(
    in_path: str = None,
    out_path: str = None,
    train_fraction: float = 0.6,
    random_seed: int = 1234,
    stratify: str = 'ea_id'
):
    
    in_path = Path(in_path)
    out_path = Path(out_path)

    if in_path.suffix == '.csv':
        full_dataset = pd.read_csv(in_path)
    elif in_path.suffix == '.parquet':
        full_dataset = pd.read_parquet(in_path)
    else:
        raise ValueError(f'Expected path to csv or parquet file for input, got {in_path}')

    # ensure determinism; we'll shuffle.
    if 'hhid' in full_dataset.columns:
        full_dataset.sort_values('hhid', inplace=True)
        full_dataset.drop(columns=['hhid'], inplace=True)
    
    # Allow stratification on multiple columns
    if '[' in stratify:
        stratify = eval(str(stratify))
        assert isinstance(stratify, list)
        full_dataset['__stratify__'] = full_dataset[stratify].astype(str).agg('-'.join, axis=1)
        stratify = '__stratify__'
        full_dataset.to_csv(out_path / 'full_with_stratifier.csv', index=False)
    train, test = train_test_split(
        full_dataset, train_size=float(train_fraction), 
        random_state=random_seed, stratify=full_dataset[stratify]
    )

    if '__stratify__' in train.columns:
        train.drop(columns=['__stratify__'], inplace=True)
    if '__stratify__' in test.columns:
        test.drop(columns=['__stratify__'], inplace=True)

    train.to_parquet(out_path / 'train.parquet')
    test.to_parquet(out_path / 'test.parquet')
    

if __name__ == '__main__':
    if False:
        _parser = argh.ArghParser()
        _parser.add_commands([main])
        _parser.dispatch()
    
    argh.dispatch_command(main)
