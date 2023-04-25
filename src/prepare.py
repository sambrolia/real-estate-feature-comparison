import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from ez_address_parser import AddressParser
import category_encoders as ce


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean the data by reformatting, normalising, and removing rows with missing assessment """

    df = parse_assessment(df)
    df = parse_ssl(df)
    df = parse_address(df)

    return df


def encode(df: pd.DataFrame, target_col) -> pd.DataFrame:
    """ Apply target encoding and one-hot encoding leaving only numeric features """

    df = target_encode(
        df, target_col, ['combined_street_info', 'ssl_a'])

    df = one_hot_encode(
        df, ['neighborhood', 'sub_neighborhood', 'StreetType', 'StreetDirection'])

    df = df.select_dtypes(include='number')

    return df


def split(df: pd.DataFrame, target_col: str):
    """ Split the data into training and testing sets """

    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def parse_assessment(df: pd.DataFrame):
    """ Remove dollar sign and commas, convert to float, remove rows with no assessment """

    df['2014_assessment'] = (
        df['2014_assessment']
        .str.replace(',', '')
        .str.replace('$', '')
        .replace('Not Available', float('nan'))
    ).astype(float)

    df = df.dropna(subset=['2014_assessment'])

    return df


def parse_ssl(df: pd.DataFrame):
    """ Split ssl into ssl_a and ssl_b

    ssl="1234    5678" -> ssl_a="1234", ssl_b="5678"
    """

    df.loc[:, 'ssl_a'] = df['ssl'].apply(lambda x: x[:4].strip())
    df.loc[:, 'ssl_b'] = df['ssl'].apply(lambda x: x[4:].strip())

    # Remove origional ssl column
    df = df.drop(columns=['ssl'])

    return df


def parse_address(df: pd.DataFrame):
    """ Parse address into new normalised feature "combined_street_info"

    1. Parse addresses using AddressParser library to create a new feature for each type of address detail: 
        e.g. new feature "StreetName" containing the parsed street names.
    2. Removes origional address column and creates new feature "combined_street_info" combining:
        StreetName, StreetType, StreetDirection, and neighborhood 
    """

    ap = AddressParser()
    for index, series in df.iterrows():
        address = series['address']
        if not isinstance(address, str):
            continue

        result = ap.parse(address)

        for token, label in result:
            if label in df.columns:
                df.at[index, label] = token
            else:
                df.insert(len(df.columns), label, '')
                df.at[index, label] = token

    df = df.drop(columns=['address'])

    df['combined_street_info'] = (
        df['StreetName']
        + df['StreetType']
        + df['StreetDirection']
        + df['neighborhood']
    )

    return df


def target_encode(df, target_col, columns_to_encode: list):
    """ Target encode columns_to_encode using target_col as the target variable """

    encoder = ce.TargetEncoder(cols=columns_to_encode)
    encoder.fit(df, df[target_col])
    df_encoded = encoder.transform(df)
    return df_encoded


def one_hot_encode(df, columns_to_encode: list):
    """ One-hot encode columns_to_encode """

    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.DataFrame(
        data=encoder.fit_transform(df[columns_to_encode]), 
        columns=encoder.get_feature_names_out(columns_to_encode), 
        index=df.index)
    
    df = pd.concat([df.drop(columns_to_encode, axis=1), encoded_df], axis=1)
    return df
