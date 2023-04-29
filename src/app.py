import pandas as pd

from flask import Flask, render_template


from config import config
from model import test_gradient_boosting, test_random_forest, top_features_gradient_boosting, top_features_random_forest, train_gradient_boosting, train_random_forest
from prepare import clean, encode, split
from visualize import plot_top_features

app = Flask(__name__)


@app.route('/')
def index():
    target_col = config['target_col']
    df = pd.read_csv(config['path'], nrows=config['nrows'])

    # Prepare the data
    df = clean(df)
    df = encode(df, target_col)
    x_train, x_test, y_train, y_test = split(df, target_col)

    # Random forest model
    rf_model = train_random_forest(x_train, y_train)
    rf_mae = test_random_forest(rf_model, x_test, y_test)
    print("Mean Absolute Error: ", rf_mae)
    rf_top_features = top_features_random_forest(
        rf_model, x_train.columns.tolist(), config['top_features'])
    print("Top " + str(config['top_features']) + " important features (Random Forest):")
    print(rf_top_features)
    rf_plot_url = plot_top_features(rf_top_features, config['plot_title'], 'forest')

    # Gradient Boosting model
    gb_model = train_gradient_boosting(x_train, y_train)
    gb_mae = test_gradient_boosting(gb_model, x_test, y_test)
    print("Gradient Boosting Mean Absolute Error: ", gb_mae)
    gb_top_features = top_features_gradient_boosting(gb_model, x_train.columns.tolist(), config['top_features'])
    print("Top " + str(config['top_features']) + " important features (Gradient Boosting):")
    print(gb_top_features)
    gb_plot_url = plot_top_features(gb_top_features, config['plot_title'], 'gradient')

    return render_template(
        "index.html",
        models=[
            ("Random Forest", rf_mae, rf_plot_url),
            ("Gradient Boosting", gb_mae, gb_plot_url)
        ]
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
