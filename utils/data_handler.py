import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import datetime as dt

from quanta.dataHandler import Dataset


def oneHotEncoding(x, labels):
    def oneHot(x):
        return x == np.asarray(labels)
    return pd.Series(oneHot(x), labels)


def degree2radian(degree):
    return 2 * degree * np.pi / 360


def harversine(initial_coord, final_coord):
    r = 6371
    initial_rads = list(map(degree2radian, initial_coord))
    final_rads = list(map(degree2radian, final_coord))
    delta_phi = final_rads[0] - initial_rads[0]
    delta_lambda = initial_rads[1] - final_rads[1]
    h = np.power(np.sin(delta_phi/2), 2) + \
        np.cos(initial_rads[0]) * np.cos(final_rads[0]) * \
        np.power(np.sin(delta_lambda/2), 2)
    return 2*r*np.arcsin(np.sqrt(h))


def logTrasnf(vector):
    return list(map(np.log, vector))


def invLogTransf(vector):
    return list(map(np.exp, vector))


def completeInverse(vector):
    return list(map(lambda x: np.exp(x**(1/7)), vector))


def meanDesnorm(value, dataset):
    """."""
    mu_y, std_y = dataset.mu_y, dataset.std_y
    value = value * std_y + mu_y
    return np.asscalar(value)


def minmaxDesnorm(value, dataset):
    """."""
    _min, _max = dataset.min_y, dataset.max_y
    value = (_max-_min)*value + _min
    return np.asscalar(value)


def desnormalize(value, dataset, kind="mean"):
    """."""
    return meanDesnorm(value, dataset) if ("mean" in kind) \
        else minmaxDesnorm(value, dataset)


def normalizeExternalData(vector, dataset, kind="mean"):
    """."""
    vector = np.asarray(vector)
    if ("mean" in kind):
        mu_x, std_x = dataset.mu_x, dataset.std_x
        return (vector - mu_x) / std_x
    else:
        _min, _max = dataset.min_x, dataset.max_x
        return (vector-_min) / (_max-_min)


def getLatLngKmeans(df):
    nclusters = 100
    batch = 10000
    kmeans = MiniBatchKMeans(n_clusters=nclusters, batch_size=batch).fit(
        df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]].sample(frac=0.7))
    return kmeans


def getDatetime(string):
    return dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def getWeekDay(date):
    return date.weekday()


def getHour(date):
    return date.hour


class CompetitionData:

    def __init__(self, frac=0.5, hotstart=False):
        self.hotstart = hotstart
        self.raw_submit = None
        self.raw_train = None
        self.mean_coords = None
        self.distance_to_center = None
        self.too_far_index = None
        self.raw_weekdays = None
        self.weekday_labels = None
        self.weekdays = None
        self.other_outliers = None
        self.relevant_cols = None
        self.class_cols = None
        self.enhanced_train = None
        self.enhanced_submit = None
        self.transf = None
        self.train_dataset = None
        self.submit_data = None
        self.normalize_method = "minmax"
        self.weekday_vars()
        self.create_cols_and_class()
        self.read_data(frac)


        if not hotstart:
            self.geo_coords_vars()
            self.kmeans = getLatLngKmeans(self.raw_train[self.too_far_index].sample(frac=0.7))
            self.another_filter()


    def read_data(self, frac):
        self.raw_submit = pd.read_csv("data/test.csv")
        self.raw_train = pd.read_csv("data/train.csv").sample(frac=frac)

    def geo_coords_vars(self):
        mean_pickup_position = self.raw_train[["pickup_latitude", "pickup_longitude"]].mean()
        mean_dropoff_position = self.raw_train[["dropoff_latitude", "dropoff_longitude"]].mean()
        self.mean_coords = pd.DataFrame([mean_pickup_position.values, mean_dropoff_position.values]).mean().values
        self.distance_to_center = self.raw_train.apply(lambda x:
                                                       (harversine([x.pickup_latitude, x.pickup_longitude],
                                                                   self.mean_coords) +
                                                        harversine([x.dropoff_latitude, x.dropoff_longitude],
                                                                   self.mean_coords)) / 2, 1).to_frame()
        self.distance_to_center.columns = ["distance"]
        distance_desc = self.distance_to_center.describe()
        self.too_far_index = self.distance_to_center.apply(lambda x: x < (np.asscalar(distance_desc.loc["mean"]) +
                                                                          3 * np.asscalar(distance_desc.loc["std"])),
                                                           1).values

    def weekday_vars(self):
        self.raw_weekdays = [0, 1, 2, 3, 4, 5, 6]
        self.weekday_labels = {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday"
        }
        self.weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def addLatLngClusters(self, df):
        clusters_pickup = self.kmeans.predict(
            df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]])
        clusters_onehot = pd.DataFrame({"clusters": clusters_pickup}).apply(
            lambda x: oneHotEncoding(x.clusters, list(range(100))), 1)
        clusters_onehot.columns = ["cluster_" + str(c) for c in clusters_onehot.columns]
        return pd.concat([df.reset_index(drop=True), clusters_onehot], axis=1)

    def addWeekday(self, df):
        temp = df.apply(lambda x: oneHotEncoding(
            getWeekDay(getDatetime(x.pickup_datetime)), self.raw_weekdays), 1)
        temp.columns = [self.weekday_labels[c] for c in temp.columns]
        return pd.concat([df, temp], axis=1)

    def addHour(self, df):
        """Add hour column."""
        df["hour"] = df.apply(lambda x: getHour(getDatetime(
                                                    x.pickup_datetime)), 1).values
        return df

    def addDistance(self, df, log=True):
        """Add distance col."""
        f = (lambda x: np.log(x + 0.00001)) if log else (lambda x: x)
        df["log_distance"] = df.apply(lambda x: f(harversine(
            [x.pickup_latitude, x.pickup_longitude],
            [x.dropoff_latitude, x.dropoff_longitude])), 1).values
        df["distance"] = df.apply(lambda x: harversine(
            [x.pickup_latitude, x.pickup_longitude],
            [x.dropoff_latitude, x.dropoff_longitude]), 1).values
        df['distance_to_center'] = df.apply(
            lambda x: np.log((harversine([x.pickup_latitude, x.pickup_longitude], self.mean_coords) + \
                              harversine([x.dropoff_latitude, x.dropoff_longitude], self.mean_coords)) / 2 + 0.000001),
            1).values
        return df

    def another_filter(self):
        y = self.raw_train[self.too_far_index][['trip_duration']]
        y_desc = y.describe()
        self.other_outliers = y.apply(lambda x: x < np.asscalar(y_desc.loc['mean']) + 3 * np.asscalar(y_desc.loc['std']),
                               1).values

    def create_cols_and_class(self):
        self.relevant_cols = ["distance", "log_distance", "hour", "passenger_count"] + self.weekdays + \
                ["cluster_" + str(i) for i in range(100)] + \
                [ "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",'distance_to_center']
        self.class_cols = [1, 1, 1] + [0 for i in self.weekdays] + [0 for i in range(100)] + \
             [1, 1, 1, 1, 1]

    def enhance(self, df, has_output=True):
        df = self.addDistance(df)
        df = self.addWeekday(df)
        df = self.addHour(df)
        df = self.addLatLngClusters(df)
        if has_output:
            df["log_duration"] = logTrasnf(df.trip_duration.values)
        return df.dropna()

    def get_enhanced_train(self):
        self.enhanced_train = pd.read_csv('data/enhanced_train.csv') if self.hotstart else self.enhance(self.raw_train)

    def get_enhanced_submit(self):
        self.enhanced_submit = pd.read_csv('data/enhanced_submit.csv') if self.hotstart else self.enhance(self.raw_submit, False)

    def set_train_dataset(self):
        if self.enhanced_train is None:
            self.get_enhanced_train()
        input_data = self.enhanced_train[self.relevant_cols]
        output_data = self.enhanced_train[["log_duration"]]
        self.transf = {'log_distance': lambda x: x ** 7}
        self.train_dataset = Dataset(input_data, output_data,
                          normalize=self.normalize_method,
                          datatypes={"input_data": self.class_cols, "output_data": [1]},
                          apply=self.transf)

    def submit_dataset(self):
        if self.enhanced_submit is None:
            self.get_enhanced_submit()
        submit_data = self.enhanced_submit[self.relevant_cols].apply(
            lambda x: normalizeExternalData(x, self.train_dataset, self.normalize_method), 1)
        for col in self.transf:
            submit_data[col] = submit_data[col].apply(self.transf[col]).values
        self.submit_data = submit_data
        return submit_data

