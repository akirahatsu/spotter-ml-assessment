import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


def clean_data(df, weight_mean, market_mean):
    df = df.copy()

    # Fix negative weights
    negative_mask = df['weight'] < 0
    df.loc[negative_mask, 'weight'] *= -1

    # Fill missing values using training-set means
    df['weight'] = df['weight'].fillna(weight_mean)
    df['market_index'] = df['market_index'].fillna(market_mean)

    return df


def main():

    # Load data


    df_train_test = pd.read_csv("/content/train-test.csv")
    df_validation = pd.read_csv("/content/validation.csv")
    df_valid_pred_temp = pd.read_csv(
        "/content/validation-predictions-template.csv"
    )
    df_dec_chart_input = pd.read_csv(
        "/content/december-chart-inputs.csv"
    )


    # Clean data


    # Calculate statistics ONLY from development data
    weight_mean = df_train_test['weight'].abs().mean()
    market_mean = df_train_test['market_index'].mean()

    df_train_test = clean_data(
        df_train_test,
        weight_mean,
        market_mean
    )

    df_validation = clean_data(
        df_validation,
        weight_mean,
        market_mean
    )



    # Internal train/test split

    train = df_train_test[
        df_train_test['date'] < '2025-09-01'
    ]

    test = df_train_test[
        df_train_test['date'] >= '2025-09-01'
    ]


    features = [
        'pickup_lat',
        'pickup_lon',
        'delivery_lat',
        'delivery_lon',
        'distance',
        'weight',
        'market_index',
        'quote_signal'
    ]


    X_train = train[features]
    Y_train = train['posted_rate']

    X_test = test[features]
    Y_test = test['posted_rate']



    # Model experiments


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


    for name, model in models.items():

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        score = r2_score(Y_test, Y_pred)

        print(f"{name}: R² = {score:.4f}")



    # Final model


    final_model = HistGradientBoostingRegressor(
        random_state=42
    )


    X_full = df_train_test[features]
    Y_full = df_train_test['posted_rate']

    final_model.fit(X_full, Y_full)


    # Validation predictions

    X_validation = df_validation[features]

    validation_pred = final_model.predict(X_validation)


    df_valid_pred_temp['predicted_rate'] = validation_pred

    df_valid_pred_temp.to_csv(
        "/content/validation_predictions.csv",
        index=False
    )


    print("validation_predictions.csv saved.")


if __name__ == "__main__":
    main()
