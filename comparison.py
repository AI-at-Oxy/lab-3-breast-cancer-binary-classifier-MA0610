#Random Forest builds decision trees using random subsets of the data and features, and then aggregates their predictions. It can capture complex relationships in the data and is often more accurate than simpler models. Although it may not be as interpretable as a linear model, it can provide feature importance scores to help understand which features are most influential in the predictions. And I specifically chose it becomes it performs well on structured tabular data like the breast cancer data. It can also capture nonlinear relationships that a simple linear model can't.


#COMMENT ON RESULTS
# The from-scratch linear model ended up with a (slightly) higher test accuracy than the Random Forest model. This may be due to the fact that the breast cancer dataset is relatively small and may not have enough complexity for the Random Forest to outperform the linear model. Additionally, the linear model may have been able to capture the underlying relationships in the data effectively, while the Random Forest may have overfitted to the training data. It's also possible that with more tuning of things like the hyperparameters or using a different ensemble method, the Random Forest could maybe achieve better performance.

import torch
from sklearn.ensemble import RandomForestClassifier
from binary_classification import load_data, train, predict, accuracy


def main():
    # Load the data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Train the from-scratch model
    w, b, _ = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)

    scratch_test_pred = predict(X_test, w, b)
    scratch_test_acc = accuracy(y_test, scratch_test_pred)


    # Random Forest model training
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Convert tensors to numpy for sklearn
    rf.fit(X_train.numpy(), y_train.numpy())
    rf_test_pred = rf.predict(X_test.numpy())

    rf_test_acc = (rf_test_pred == y_test.numpy()).mean()

    # ===============================
    # 3. Compare Results
    # ===============================
    print("From-Scratch Model Test Accuracy:", round(scratch_test_acc, 4))
    print("Random Forest Test Accuracy:", round(rf_test_acc, 4))

if __name__ == "__main__":
    main()