# spotter-ml-assessment

# Issues i find out from Data

$ Negative Weight in Validation and Train_Test it is noticable in Minimum

          weight
          count	47700.000000
          mean	31028.844004
          std	9391.440620
          min	-47500.000000
          25%	25800.000000
          50%	31436.500000
          75%	37018.000000
          max	47500.000000
          
          dtype: float64
i multiplied them to -1 reasoning where mean and std of negative volues and positve volues were similar i decided it is typing Error


$ Weight and Market_index Missing data Train_test and Validation

        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 48000 entries, 0 to 47999
        Data columns (total 14 columns):
         #   Column        Non-Null Count  Dtype  
        ---  ------        --------------  -----  
         0   load_id       48000 non-null  object 
         1   pickup        48000 non-null  object 
         2   delivery      48000 non-null  object 
         3   pickup_lat    48000 non-null  float64
         4   pickup_lon    48000 non-null  float64
         5   delivery_lat  48000 non-null  float64
         6   delivery_lon  48000 non-null  float64
         7   distance      48000 non-null  float64
         8   equipment     48000 non-null  object 
         9   weight        47700 non-null  float64
         10  date          48000 non-null  object 
         11  market_index  47626 non-null  float64
         12  quote_signal  48000 non-null  float64
         13  posted_rate   48000 non-null  float64
        dtypes: float64(9), object(5)
        memory usage: 5.1+ MB
        
deal with it using avarage volues 


## Splitting data 
instead of splitting  test == 10% train == 90 i splitted them based on time data 

      train = df_train_test[df_train_test['date'] < '2025-09-01']
      test  = df_train_test[df_train_test['date'] >= '2025-09-01']

## Model i decided use HistGradientBoostingRegressor

          r2_score
          
          LinearRegression              : 0.8133237726803371
          RandomForestRegressor         : 0.8092938692850085
          HistGradientBoostingRegressor : 0.8201703056067264 


# Instead of changing string volue into numeric blindly i deleted them reason numeric volue = [1,2,3 .. ] 
# and string volues are Richmond', 'Philadelphia', 'Hartford', 'Dallas' 
# Reasoning is from 1 to 2 there is single unit change and from 2 to 3 but from Richmond to Philadelphia or from Hartford to Dallas change is not mesurable 
# like a numerical distence that is why to prevent from extra noise i decided not to use them

