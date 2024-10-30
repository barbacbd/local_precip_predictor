from math import sqrt
from statistics import mean
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from mrmr import mrmr_classif


def _parse_date(date_str):
    '''
    Parse the date out of the string in the format %Y-%M-%D HH:mm:ss
    '''
    date_split = date_str.split()[0]
    year, month, day = date_split.split("-")
    return year, month, day


def parse_daily_values_by_month(daily_value_df):
    '''
    Parse the monthly averages for each year. 

    - temperature_2m_max
    - temperature_2m_min
    - temperature_2m_mean
    - apparent_temperature_max
    - apparent_temperature_min
    - apparent_temperature_mean
    - precipitation_sum
    - rain_sum
    - snowfall_sum

    The following units are used:
    - temperature_unit: fahrenheit
    - wind_speed_unit: mph
    - precipitation_unit: inch

    :param daily_value_df: Dataframe containing the weather data for a period of time.
    :return: dictionary year -> month -> average values
    '''
    data_by_month = {}
    for index, row in daily_value_df.iterrows():

        year, month, day = _parse_date(str(row.iloc[0]))
        if year not in data_by_month:
            data_by_month[year] = {}
        if month not in data_by_month[year]:
            data_by_month[year][month] = {
                # "temperature_2m_max": [],
                # "temperature_2m_min": [],
                # "temperature_2m_mean": [],
                # "apparent_temperature_max": [],
                # "apparent_temperature_min": [],
                # "apparent_temperature_mean": [],
                # "precipitation_sum": [],
                # "rain_sum": [],
                "snowfall_sum": [],
            }
        
        # if row.iloc[1] is not None:
        #     data_by_month[year][month]["temperature_2m_max"].append(row.iloc[1])
        # if row.iloc[2] is not None:
        #     data_by_month[year][month]["temperature_2m_min"].append(row.iloc[2])
        # if row.iloc[3] is not None:
        #     data_by_month[year][month]["temperature_2m_mean"].append(row.iloc[3])
        # if row.iloc[4] is not None:
        #     data_by_month[year][month]["apparent_temperature_max"].append(row.iloc[4])
        # if row.iloc[5] is not None:
        #     data_by_month[year][month]["apparent_temperature_min"].append(row.iloc[5])
        # if row.iloc[6] is not None:
        #     data_by_month[year][month]["apparent_temperature_mean"].append(row.iloc[6])
        # if row.iloc[7] is not None:
        #     data_by_month[year][month]["precipitation_sum"].append(row.iloc[7])
        # if row.iloc[8] is not None:
        #     data_by_month[year][month]["rain_sum"].append(row.iloc[8])
        if row.iloc[9] is not None:
            data_by_month[year][month]["snowfall_sum"].append(row.iloc[9])

    averages = {}
    for year, yearly_value in data_by_month.items():
        for month, monthly_value in yearly_value.items():
            index = f"{year}-{int(month)}"

            for key in monthly_value:
               mean_value = mean(monthly_value[key])
               averages[index] = [mean_value, int(mean_value > 0)]
    
    df = pd.DataFrame.from_dict(averages, orient='index', columns=["snowfall_sum", "snowed"])

    return df


def combine_data(dfs):
    '''
    Combine the dataframes. Each dataframe should be cut down to be the
    same size as the smallest dataframe to avoid issues/missing values.
    '''
    smallest_df_len = min([len(df) for df in dfs])
    combined_df = dfs[0].head(smallest_df_len)

    for df in dfs[1:]:
        sliced_df = df.head(smallest_df_len)
        combined_df = pd.concat([combined_df, sliced_df], axis=1, join='outer')
    
    return combined_df


def find_features_mrmr(dataset, outputs, K=None, **kwargs):
    """Select K features using the MRMR feature selection algorithm.

    :param dataset: Pandas dataframe. The dataframe should NOT include the output column.
    :param outputs: Output column from the original dataset
    :param K: number of features. When None, make a guess at the optimal number of features by 
    using the mean of the relevance scores to find the features.
    """
    print(f"init k = {K}")
    _k = K
    if _k is None or _k <= 0:
        _k = dataset.shape[1]

        if 0 >= _k > dataset.shape[1]:
            print("on no")
            return None

    print(f"end k = {_k}")
    selected_features, relevance, _ = mrmr_classif(X=dataset, y=outputs, K=_k, return_scores=True)
    print(selected_features)
    print(relevance)

    if K is not None and K > 0: 
        print(selected_features[:_k])
        return selected_features

    # Find the number of values where the relevance is above the mean. These are NOT
    # the values that we will return. Instead, use this number as K
    r = relevance.to_frame(name="relevance")
    m = r['relevance'].mean()
    r.sort_values(by=['relevance'], ascending=False, inplace=True)
    _k = len(r[r.relevance > m].index.tolist())

    print(selected_features[:_k])

    return selected_features[:_k]


def find_features_f1_scores(dataset, outputs, precision=1.0, **kwargs):
    """Attempt to find the optimal features used for training 
    model(s). This is merely an estimation, but the optimal number of features is
    the least amount of features required to meet a F1 Score of `precision`. 

    :param dataset: Pandas dataframe. The dataframe should NOT include the output column.
    :param outputs: Output column from the original dataset
    :param precision: Minimum F1 Score used for calculations. When F1 scores are below this
    value, the optimal number of features has not been achieved.
    """

    # Create a deep copy so that changes are not reflected to the original
    # dataframe.
    df = dataset.copy(True)

    forest = RandomForestClassifier(n_jobs=1, random_state=42)
    forest.fit(df, outputs)

    f1Scores = []
    feats = []

    while df.shape[1] > 0:
        feature_importances = forest.feature_importances_

        y_pred = forest.predict(df)
        f1 = f1_score(outputs, y_pred, average='micro')
        f1Scores.append(f1)
        feats.append(df.columns)

        if df.shape[1] == 1:
            break

        least_important_idx = feature_importances.argmin()

        df.drop(df.columns[least_important_idx], axis=1, inplace=True)
        forest.fit(df, outputs)

    # number of features that should be used.
    numFeaturesToUse = 0

    # F1 Values should drop as the number of features is reduced eventually leading
    # towards 0, reverse this list so that the smallest values are first.
    f1Scores.reverse()

    # get the minimal number of features until we have reached the optimal precision
    for f1s in f1Scores:
        if f1s >= precision:
            break
        numFeaturesToUse += 1

    print(numFeaturesToUse)
    print(feats[-numFeaturesToUse].tolist())

    return feats[-numFeaturesToUse].tolist()


def correct_data(df, droppable=[]):
    '''
    Alter the dataframe to remove categorical data. When item(s) are provided
    via the `droppable` argument, those columns will be removed from the dataframe too.
    '''
    # report the output values. This can be used as a prediction
    # value or a training data outcome
    output = df["snowed"]

    # Drop the output from the Dataframe, leaving the only data left as
    # the dataset to train.
    df.drop(labels=["snowed", "snowfall_sum", "AMO", "ENSO"], axis=1,inplace=True)
    df.fillna(0, inplace=True)

    return df, output


def create_model(df, **kwargs):
    """

    :kwargs:
    - numEpochs - number of epochs to be run during model training
    - batchSize - batch size input for model training
    - other - please see the feature selection algorithms in `features.py` for more information
    as inputs to the algorithms.
    """

    # filter out the output/winner and a few categorical columns
    trainDF, trainOutput = correct_data(df, droppable=[])

    # use the default values for feature selection (when applicable).
    # features = find_features_f1_scores(trainDF, trainOutput, **kwargs)
    features = find_features_mrmr(trainDF, trainOutput, **kwargs)
    # return

    # only keep the features that we selected
    trainDF = trainDF[features]
    _, numLabels = trainDF.shape

    model = Sequential()
    # Create a model for a neural network with 3 layers
    # According to a source online, ReLU activation function is best for layers
    # except the output layer, that layer should use Sigmoid. This is the case for
    # performance reasons.
    level = min([numLabels, 16])
    model.add(Dense(level, input_shape=(numLabels,), activation='relu'))
    while level > 0:
        level = int(sqrt(level))
        if level <= 1:
            # Looking for a single value 0 or 1 for the output
            model.add(Dense(1, activation='sigmoid'))
            level = 0
        else:
            model.add(Dense(level, activation='sigmoid'))

    # create the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # tensorflow requires data in a specific format. convert this the expected format
    dfTensor = tf.convert_to_tensor(trainDF.to_numpy())
    outputTensor = tf.convert_to_tensor(trainOutput.to_numpy())

    model.fit(dfTensor, outputTensor, epochs=kwargs["num_epochs"], 
              batch_size=kwargs["batch_size"])
    _, accuracy = model.evaluate(dfTensor,  outputTensor)

    # attempt to save the model
    model.save("snowfall.keras")

    return model