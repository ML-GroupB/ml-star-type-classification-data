from sklearn import impute

# SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def simple_imputer(data, strategy="mean"):
    """
    Imputation transformer for completing missing values to data.

    :param data: Array like list.
    :param strategy: Imputation strategy. Check sklearn.impute.SimpleImputer
    :return: Data with imputer on X.
    """
    imp = impute.SimpleImputer(strategy=strategy)
    return data.fit(imp)


# OrdinalEncoder
def ordinal_encoder(data):
    """
    Encode text data to numbers.
    Example: [[0.],[1.].[2.],[0.]]

    :param data: Array like list
    :return: Data with encoded text.
    """
    ordenc = OrdinalEncoder()
    return ordenc.fit_transform(data)


# OneHotEncoder
def one_hot_encoder(data):
    """
    Encode text data to numbers.
    Example: [[0. 0. 1.],[0. 1. 0.],[1. 0. 0.],[1. 0. 0.]]

    :param data:
    :return:
    """
    onehot = OneHotEncoder()
    data = onehot.fit_transform(data)
    return data.toarray()


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    from sklearn.compose import ColumnTransformer

    X = [1, 3, 8, 54, 52, 43]
    Y = ["Liczby"]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = ["Length", "Height"]
    categorical_features = ["Type"]

    full_pipeline = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    data_prepared = full_pipeline.fit_transform(X, Y)

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    """
    First Method
    """

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(data_prepared, Y)

    lin_predictions = lin_reg.predict(data_prepared)

    # lin_predictions
    print(cross_val_score(lin_reg, data_prepared, Y))


    # DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(data_prepared, Y)

    tree_predictions = tree_reg.predict(data_prepared)

    # tree_predictions
    print(cross_val_score(tree_reg, data_prepared, Y))


    # RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(data_prepared, Y)

    forest_predictions = forest_reg.predict(data_prepared)

    # forest_predictions
    print(cross_val_score(forest_reg, data_prepared, Y))


