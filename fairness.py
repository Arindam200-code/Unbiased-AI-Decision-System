from fairlearn.metrics import demographic_parity_difference

def check_bias(y_true, y_pred, sensitive_feature):

    return demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )