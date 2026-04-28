from fairlearn.metrics import demographic_parity_difference

def check_bias(y_true, predictions, sensitive_feature):

    return demographic_parity_difference(
        y_true=y_true,
        y_pred=predictions,
        sensitive_features=sensitive_feature
    )