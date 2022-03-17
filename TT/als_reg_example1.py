import logging

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def als_reg_driver():
    pass
def lin_reg(X, Y, n_test):
    logger = logging.getLogger()
    # Split the data into training/testing sets
    X_train = X[:-n_test]
    X_test = X[-n_test:]

    # Split the targets into training/testing sets
    y_train = Y[:-n_test]
    y_test = Y[-n_test:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(X_test)

    # The coefficients
    logger.info(f"Coefficients: {regr.coef_}\n")
    # The mean squared error
    logger.info(f"Mean squared error: { mean_squared_error(y_test, diabetes_y_pred)}")
    # The coefficient of determination: 1 is perfect prediction
    logger.info(f"Coefficient of determination: {r2_score(y_test, diabetes_y_pred)}")


if __name__ == '__main__':
    LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT)

    # Load the diabetes dataset
    X, Y = datasets.load_diabetes(return_X_y=True)

    # Params
    n_test = 20
    ###############

    lin_reg(X=X, Y=Y, n_test=n_test)
