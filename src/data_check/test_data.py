import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Test that the dataset contains all and only the expected columns (variables) based on their names.
    :param data: The dataframe with the dataset to be checked.
    :type data: pandas.DataFrame
    """
    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):
    """
    Test that the values for variable `neighbourhood_group` are the expected ones.
    :param data: The dataset.
    :type data: pandas.DataFrame
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC.
    :param data: The dataset.
    :type data: pandas.DataFrame
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different from that of the reference dataset
    :param data: The dataset to be tested.
    :type data: pandas.Dataframe
    :param ref_data: The reference dataset that `data` has to be tested against.
    :type ref_data: pandas.DataFrame
    :param kl_threshold: The threshold for the Kullback-Leibler divergence between the two datasets, such that if the
    threshold is met or exceeded, the test fails.
    :type kl_threshold: float
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """
    Verify that the number of samples in the dataset is sound.
    :param data: The given dataset.
    :type data: pandas.DataFrame
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    """
    Test that all the values of the `price` variable are within a given range.
    :param data: The dataset.
    :type data: pandas.DataFrame
    :param min_price: The lower bound for the given range.
    :type min_price: float
    :param max_price: The upper-bound for the given range.
    :type max_price: float
    """
    assert data['price'].between(min_price, max_price).all()
