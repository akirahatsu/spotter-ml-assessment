import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score



# CLEAN DATA


def clean_data(df, weight_mean, market_index_mean):
    df = df.copy()

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Fix negative weights
    negative_mask = df['weight'] < 0
    df.loc[negative_mask, 'weight'] *= -1

    # Fill missing weight using training mean
    df['weight'] = df['weight'].fillna(weight_mean)

    # Fill missing market_index using training mean, if column exists or create it
    if 'market_index' not in df.columns:
        df['market_index'] = np.nan # Create column if it doesn't exist
    df['market_index'] = df['market_index'].fillna(market_index_mean)

    return df


# FEATURE ENGINEERING


def create_features(df):
    df = df.copy()

    # Date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek

    # Remove original date
    df = df.drop(columns=['date'])

    # Remove ID
    if 'load_id' in df.columns:
        df = df.drop(columns=['load_id'])

    # One-hot encode categorical columns
    categorical_columns = [
        'pickup',
        'delivery',
        'equipment'
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    return df



def main():


    # LOAD DATA

    df_train_test = pd.read_csv(
        "/content/train-test.csv"
    )

    df_validation = pd.read_csv(
        "/content/validation.csv"
    )

    df_valid_pred_temp = pd.read_csv(
        "/content/validation-predictions-template.csv"
    )

    df_dec_chart_input = pd.read_csv(
        "/content/december-chart-inputs.csv"
    )



    # TRAINING-SET STATISTICS


    weight_mean = (
        df_train_test['weight']
        .abs()
        .mean()
    )

    market_index_mean = (
        df_train_test['market_index']
        .mean()
    )


    # CLEAN DATA


    df_train_test = clean_data(
        df_train_test,
        weight_mean,
        market_index_mean
    )

    df_validation = clean_data(
        df_validation,
        weight_mean,
        market_index_mean
    )

    df_dec_chart_input = clean_data(
        df_dec_chart_input,
        weight_mean,
        market_index_mean
    )



    # INTERNAL TIME-BASED VALIDATION


    train = df_train_test[
        df_train_test['date'] < '2025-09-01'
    ].copy()

    test = df_train_test[
        df_train_test['date'] >= '2025-09-01'
    ].copy()



    # Separate target BEFORE feature engineering


    Y_train = train['posted_rate']
    Y_test = test['posted_rate']



    # Combine train/test temporarily so that categorical
    # encoding creates exactly the same columns


    combined = pd.concat(
        [
            train.drop(columns=['posted_rate']),
            test.drop(columns=['posted_rate'])
        ],
        axis=0
    )

    combined_features = create_features(combined)


    # Split back
    X_train = combined_features.iloc[:len(train)]
    X_test = combined_features.iloc[len(train):]


    # MODEL EXPERIMENTS


    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42
        )
    }


    print("\nMODEL RESULTS")
    print("=" * 40)


    for name, model in models.items():

        model.fit(
            X_train,
            Y_train
        )

        Y_pred = model.predict(
            X_test
        )

        score = r2_score(
            Y_test,
            Y_pred
        )

        print(
            f"{name}: R² = {score:.4f}"
        )



    # FINAL MODEL


    print("\nTraining final model...")


    # Target
    Y_full = df_train_test['posted_rate']


    # Features
    X_full_raw = df_train_test.drop(
        columns=['posted_rate']
    )

    X_full = create_features(
        X_full_raw
    )


    # Final model
    final_model = HistGradientBoostingRegressor(
        random_state=42
    )


    final_model.fit(
        X_full,
        Y_full
    )



    # VALIDATION PREDICTIONS


    X_validation = create_features(
        df_validation
    )



    X_validation = X_validation.reindex(
        columns=X_full.columns,
        fill_value=0
    )


    validation_pred = final_model.predict(
        X_validation
    )


    df_valid_pred_temp['predicted_rate'] = (
        validation_pred
    )


    df_valid_pred_temp.to_csv(
        "/content/validation_predictions.csv",
        index=False
    )


    print(
        "\nvalidation_predictions.csv saved."
    )



    # DECEMBER PREDICTIONS


    X_december = create_features(
        df_dec_chart_input
    )


    # IMPORTANT:
    # December must have exactly the same feature
    # columns as training.

    X_december = X_december.reindex(
        columns=X_full.columns,
        fill_value=0
    )


    december_pred = final_model.predict(
        X_december
    )


    df_dec_chart_input['predicted_rate'] = (
        december_pred
    )


    df_dec_chart_input.to_csv(
        "/content/december-chart-inputs.csv",
        index=False
    )


    print(
        "december-chart-inputs.csv updated."
    )


    # SHOW DECEMBER PREDICTIONS


    print("\nDECEMBER PREDICTIONS")
    print("=" * 40)

    print(
        df_dec_chart_input[
            ['date', 'predicted_rate']
        ]
    )


if __name__ == "__main__":
    main()
