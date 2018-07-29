from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose


def time_series_decomposition(series, log_scale, frequency, two_sided, plot=False, only_resid=False):
    if log_scale:
        series = np.log(series)

    result = seasonal_decompose(series.tolist(), model='multiplicative', freq=frequency, two_sided=two_sided)

    if plot:
        result.plot()
        plt.show()

    if only_resid:
        # return only residuals
        return result.resid
    else:
        return result


def time_series_annotation_optimization(data, col_time_series, col_date, col_label, frequency, left_tail_multiplier,
                                        right_tail_multiplier, log_scale=True, two_sided=True, plot=False,
                                        metrics=False):
    # Decomposition of the closing values
    result = time_series_decomposition(data[col_time_series], log_scale, frequency, two_sided, plot)

    residuals = pd.Series(result.resid)
    seasonal = pd.Series(result.seasonal)
    trend = pd.Series(result.trend)

    if plot:
        sns.distplot(residuals[residuals > 0])

    # Define parameters for the "confidence interval" that will define annotation labels of the time series
    mean_open = residuals[residuals > 0].mean()
    std_open = residuals[residuals > 0].std()

    # Annotation Interval
    interval_max, interval_min = mean_open + std_open - std_open * right_tail_multiplier, \
                                 mean_open - std_open + std_open * left_tail_multiplier
    print("interval: [{}, {}]".format(interval_min, interval_max))

    # Select periods out of the given intervals with residuals
    period_1_idx = residuals[(residuals < interval_min)].index.tolist()
    period_2_idx = residuals[(residuals < interval_max) & \
                             (residuals > interval_min)].index.tolist()
    period_3_idx = residuals[(residuals > interval_max)].index.tolist()

    # annotate data
    data_labeled = data.copy()
    data_labeled[col_label] = 'unknown'
    data_labeled.ix[period_1_idx, col_label] = 'crisis'
    data_labeled.ix[period_2_idx, col_label] = 'standard'
    data_labeled.ix[period_3_idx, col_label] = 'pre_crisis'

    # add residuals, seasonal and trend to the data
    data_labeled['residuals'] = residuals
    data_labeled['seasonal'] = seasonal
    data_labeled['trend'] = trend

    if metrics:
        metrics = annotation_evaluation(data_labeled, col_date, col_label)
        return metrics, data_labeled
    else:
        return data_labeled


def annotation_evaluation(data_annotated, date_column, label_column):
    metrics = {}
    ## Metrics for crisis period: is the month after the crisis labeled as crisis?
    # crisis May 1962
    perc_c_1 = check_label(data_annotated, 1962, 6, 'crisis', date_column, label_column)

    # crisis Oct 1987
    perc_c_2 = check_label(data_annotated, 1987, 11, 'crisis', date_column, label_column)

    # crisis July 1990
    perc_c_3 = check_label(data_annotated, 1990, 8, 'crisis', date_column, label_column)

    # crisis Oct 2002
    perc_c_4 = check_label(data_annotated, 2002, 11, 'crisis', date_column, label_column)

    # crisis Sep 2008
    # perc_c_5 = check_label(data_annotated, 2008, 10, 'crisis', date_column, label_column)

    ## Metrics for pre-crisis period: is the month before the crisis labeled as pre-crisis?
    perc_p_1 = check_label(data_annotated, 1962, 3, 'pre_crisis', date_column, label_column)
    perc_p_2 = check_label(data_annotated, 1987, 8, 'pre_crisis', date_column, label_column)
    perc_p_3 = check_label(data_annotated, 1990, 5, 'pre_crisis', date_column, label_column)
    perc_p_4 = check_label(data_annotated, 2002, 8, 'pre_crisis', date_column, label_column)
    # perc_p_5 = check_label(data_annotated, 2008, 7, 'pre_crisis', date_column, label_column)

    ## Metrics for standard period
    perc_s_1 = check_label(data_annotated, 1991, 4, 'standard', date_column, label_column)
    # perc_s_2 = check_label(data_annotated, 2006, 9, 'standard', date_column, label_column)
    perc_s_3 = check_label(data_annotated, 1984, 6, 'standard', date_column, label_column)
    perc_s_4 = check_label(data_annotated, 1978, 9, 'standard', date_column, label_column)
    perc_s_5 = check_label(data_annotated, 1959, 8, 'standard', date_column, label_column)

    metrics['perc_c_1'] = perc_c_1
    metrics['perc_c_2'] = perc_c_2
    metrics['perc_c_3'] = perc_c_3
    metrics['perc_c_4'] = perc_c_4
    # metrics['perc_c_5'] = perc_c_5
    metrics['perc_p_1'] = perc_p_1
    metrics['perc_p_2'] = perc_p_2
    metrics['perc_p_3'] = perc_p_3
    metrics['perc_p_4'] = perc_p_4
    # metrics['perc_p_5'] = perc_p_5
    metrics['perc_s_1'] = perc_s_1
    # metrics['perc_s_2'] = perc_s_2
    metrics['perc_s_3'] = perc_s_3
    metrics['perc_s_4'] = perc_s_4
    metrics['perc_s_5'] = perc_s_5
    return metrics


def check_label(data, year, month, annotation, date_column, label_column):
    correct_label = len(data[(data[date_column].dt.year == year) & (data[date_column].dt.month == month) & \
                             (data[label_column] == annotation)])
    total_available = len(data[(data[date_column].dt.year == year) & (data[date_column].dt.month == month)])
    if total_available == 0:
        return 0
    else:
        percent = correct_label * 100 / total_available
        return percent


def complete_crisis_annotation(data, annotation_column, thrs_btw_crisis_i, thrs_btw_crisis_g, thrs_btw_precrisis_i,
                               thrs_btw_precrisis_g):
    # Initialize counters
    counter_crisis = 0
    counter_precrisis = 0
    counter_standard = 0
    first_precrisis = 0
    last_precrisis = 0
    first_crisis = 0
    last_crisis = 0
    label = 'standard'

    for i, day in data.iterrows():

        counter_crisis += 1 if day[annotation_column] == 'crisis' else 0
        counter_precrisis += 1 if day[annotation_column] == 'pre_crisis' else 0
        counter_standard += 1 if day[annotation_column] == 'standard' else 0

        # we start the time series annotation correction by assigning the correct label at the beginning
        if label == 'standard':
            if day[annotation_column] == 'crisis':
                label = 'crisis'
                first_crisis = i
                last_crisis = i

            elif day[annotation_column] == 'pre_crisis':
                label = 'pre_crisis'
                first_precrisis = i
                last_precrisis = i


        # case that in x days/months/year more than one crisis event => roll-back "crisis"
        # label to all events in between
        elif label == 'crisis':
            if counter_crisis == 1:
                if i - first_crisis > thrs_btw_crisis_i:
                    data.loc[first_crisis, annotation_column] = 'standard'
                    counter_crisis = 0
                    counter_precrisis = 0
                    label = 'pre_crisis'

            elif counter_crisis > 1:
                if day[annotation_column] == 'crisis':

                    if i - last_crisis < thrs_btw_crisis_g:
                        if i - first_crisis < thrs_btw_crisis_i:
                            data.loc[last_crisis:i, annotation_column] = 'crisis'
                            last_crisis = i
                            counter_precrisis = 0
                            last_precrisis = 0
                            first_precrisis = 0
                    else:
                        first_crisis = i
                        last_crisis = i
                        counter_crisis = 1

                elif day[annotation_column] == 'pre_crisis':
                    label = 'pre_crisis'
                    counter_crisis = 0
                    first_precrisis = i
                    last_precrisis = i

        # case that in x days/months/year more than one pre-crisis event => roll-back "pre_crisis"
        # label to all events in between
        elif label == 'pre_crisis':
            if counter_crisis == 1:
                if day[annotation_column] == 'crisis':
                    label = 'crisis'
                    first_crisis = i
                    last_crisis = i

            elif counter_precrisis > 1:
                if day[annotation_column] == 'pre_crisis':
                    if i - last_precrisis < thrs_btw_precrisis_g:
                        if i - first_precrisis < thrs_btw_precrisis_i:
                            data.loc[last_precrisis:i, annotation_column] = 'pre_crisis'
                            last_precrisis = i
                    else:
                        data.loc[first_precrisis:last_precrisis, annotation_column] = 'standard'
                        counter_precrisis = 1
                        first_precrisis = i
                        last_precrisis = i

    return data


def gradient_feature_calculation(data, frequency, month_period):
    # Extract the value of the first event index to start iteration (considering drop of data due
    # to seasonal decomposition (as large as the frequency choosen)
    first = data.head(1).index[0] + frequency

    events_features = []
    for event in range(first, frequency + len(data)):
        features = {}

        resid = data.loc[event]['residuals']
        features['residual'] = resid

        day = data.loc[event]['day']
        month = data.loc[event]['month']
        year = data.loc[event]['year']

        for period in month_period:

            # Obtain the reference date to calculate the gradient
            ref_date = date(year, month, day) + relativedelta(months=-period)
            n_month = ref_date.month
            n_year = ref_date.year

            # since there may be the case that there is missing data for a given day
            # we get all data for the given month and extract the value of the first day available
            try:
                ref = data[(data['year'] == n_year) & (data['month'] == n_month)]['residuals'].values[0]
            except Exception as exp:
                print("no data for year {} and month {}: {}".format(n_year, n_month, exp))
                ref = np.nan

            features['gradient_' + str(period)] = resid - ref

        events_features.append(features)

    # organize results into a dataframe
    df_events_features = pd.DataFrame.from_records(events_features)
    df_events_features = df_events_features.loc[:, ('residual', 'gradient_2', 'gradient_4', 'gradient_8',
                                                    'gradient_16', 'gradient_32')]

    return df_events_features


def match_features_to_events(events, features, frequency, features_scale, x_time_col, col_time_series):
    # Assign to featured events the previous indexes for correct matching
    first = events.head(1).index[0] + frequency
    end = len(events) + frequency
    features.index = events.loc[first:end].index

    # Add date column to features for label matching in training set
    df_features = features.merge(pd.DataFrame(events.loc[:, (col_time_series, x_time_col)]), left_index=True,
                                 right_index=True)

    # Drop NaN values (if any gradient could not have values to be calculated)
    df_features = df_features.dropna()

    # Scale values to be parsed through the model (mean = 0, std=1)
    for column in features_scale:
        scaler = StandardScaler()
        df_features[column] = scaler.fit_transform(df_features[column])

    return df_features


def build_feature_vector_from_time_series(data, log_scale, frequency, two_sided, plot, only_resid, month_period,
                                          col_time_series, col_date, x_time_col, features_scale):
    # Decompose time series and extract only residuals
    data['residuals'] = time_series_decomposition(data[col_time_series], log_scale, frequency, two_sided, plot,
                                                  only_resid)

    # Extract gradients as features for each event
    # Drop nan due to decomposition
    data_ = data.dropna()

    # Extract day, month and year out of the date for easier gradient calculation
    data_['day'] = data_[col_date].dt.day
    data_['month'] = data_[col_date].dt.month
    data_['year'] = data_[col_date].dt.year

    # Calculate gradients for event's residuals and past events as features for our time series
    df_events_features = gradient_feature_calculation(data_, frequency, month_period)
    feature_vector = match_features_to_events(data_, df_events_features, frequency, features_scale, x_time_col,
                                              col_time_series)

    return feature_vector
