import base64
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from ez_address_parser import AddressParser
import category_encoders as ce
from config import config

app = Flask(__name__)


@app.route('/')
def index():
    # Load and preprocess your data
    df = pd.read_csv(config['path'], nrows=config['nrows'])

    df = preprocess_data(df)

    # Train the model and find the most important feature
    target_col = "2014_assessment"

    # Combine street info and Apply target encoding
    df['combined_street_info'] = \
        df['StreetName'] + ' ' + \
        df['StreetType'] + ' ' + \
        df['StreetDirection'] + ' ' + \
        df['neighborhood']

    df = target_encode(df, target_col, 'combined_street_info')

    # Apply one-hot encoding to specified categorical columns
    one_hot_cols = ['neighborhood', 'sub_neighborhood', 'StreetType', 'StreetDirection']
    df = one_hot_encode(df, one_hot_cols)

    # Remove remaining categorical columns so only numeric columns remain
    df = df.select_dtypes(include='number')

    model = train_random_forest(df, target_col)

    # Make sure to pass the correct feature_names list
    feature_names = df.drop(target_col, axis=1).columns.tolist()
    important_feature = find_most_important_feature(model, feature_names)

    print(f"Important feature: {important_feature}")

    # Plot the most important feature
    img = BytesIO()
    plot_most_important_feature(df, important_feature, target_col, "Most Important Feature vs Assessment")
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Train the RandomForestRegressor model on the filtered dataset
    filtered_model = train_random_forest(df, target_col)

    # Find the most important features and check if StreetName has higher importance than use_code
    filtered_feature_names = df.drop(target_col, axis=1).columns.tolist()
    filtered_important_feature = find_most_important_feature(filtered_model, filtered_feature_names)
    print(f"Important feature (filtered dataset): {filtered_important_feature}")

    importances = filtered_model.feature_importances_
    street_name_importance = importances[filtered_feature_names.index('combined_street_info')]
    use_code_importance = importances[filtered_feature_names.index('use_code')]

    print(f"combined_street_info importance: {street_name_importance:.4f}")
    print(f"use_code importance: {use_code_importance:.4f}")

    if street_name_importance > use_code_importance:
        print("In the filtered dataset, combined_street_info has higher importance than use_code.")
    else:
        print("In the filtered dataset, use_code still has higher importance than StreetName.")

    return render_template("index.html", plot_url=plot_url)


def target_encode(df, target_col, column_to_encode):
    encoder = ce.TargetEncoder(cols=[column_to_encode])
    encoder.fit(df, df[target_col])
    df_encoded = encoder.transform(df)
    return df_encoded


def one_hot_encode(df, columns_to_encode):
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(df[columns_to_encode])
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)
    df = pd.concat([df.drop(columns_to_encode, axis=1), encoded_df], axis=1)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parseAssessment(df)
    df = parseSSL(df)
    df = parseAddress(df)

    return df


def parseAssessment(df: pd.DataFrame):
    df['2014_assessment'] = df['2014_assessment'].str.replace(',', '').str.replace('$', '')
    df['2014_assessment'] = df['2014_assessment'].replace('Not Available', float('nan')).astype(float)
    df = df.dropna(subset=['2014_assessment'])

    return df


def parseSSL(df: pd.DataFrame):
    df['ssl_a'] = df['ssl'].apply(lambda x: x[:4].strip())
    df['ssl_b'] = df['ssl'].apply(lambda x: x[4:].strip())
    df = df.drop(columns=['ssl'])

    return df


def parseAddress(df: pd.DataFrame):
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

    return df


def train_random_forest(df: pd.DataFrame, target_col: str) -> RandomForestRegressor:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")

    return model


def find_most_important_feature(model: RandomForestRegressor, feature_names: list) -> str:
    importances = model.feature_importances_
    idx = importances.argmax()

    return feature_names[idx]


def plot_most_important_feature(df: pd.DataFrame, important_feature: str, target_col: str, title: str) -> None:
    plt.scatter(df[important_feature], df[target_col], alpha=0.5)
    plt.xlabel(important_feature)
    plt.ylabel(target_col)
    plt.title(title)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
