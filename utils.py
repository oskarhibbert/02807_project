# Function to categorise delay times
def categorise_delay(delay):
    if delay <= 15:          # Minor delay
        return 0
    elif 0 < delay <= 60:    # Moderate delay
        return 1
    elif 30 < delay <= 120:  # Significant delay
        return 2
    else:                    # Severe delay
        return 3
    


# Function to categorise weather conditions relative to historical weather conditions AT LOCATION
def categorise_weather(df, weather_col, station_col):
    station_stats = df.groupby(station_col)[weather_col].agg(['mean', 'std']).reset_index()
    
    df = df.merge(station_stats, on=station_col, how='left', suffixes=('', '_stats'))
    
    conditions = [
        (df[weather_col] < df['mean'] - df['std']),  # Much Lower than average
        (df[weather_col] < df['mean']),              # Lower than average
        (df[weather_col] < df['mean'] + df['std']),  # Higher than average
        (df[weather_col] >= df['mean'] + df['std'])  # Much Higher than average
    ]
    
    categories = [-2, -1, 1, 2]

    df[f'{weather_col}_category'] = np.select(conditions, categories, default='Average')
    
    df.drop(['mean', 'std'], axis=1, inplace=True)
    
    return df