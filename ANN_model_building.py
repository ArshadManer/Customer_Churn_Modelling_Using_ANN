import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

churn_data = pd.read_csv('Processed_data.csv',index_col='RowNumber')

def prepare_data(data):
    # Assuming 'Exited' is the target variable
    X = data.drop(['Exited'], axis=1)
    y = data.Exited
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Feature Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_ann():
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Adding the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

def evaluate_model(model, X, y):
    score, acc = model.evaluate(X, y, batch_size=10)
    print('Score:', score)
    print('Accuracy:', acc)

    # Predicting the results
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Classification Report
    print('Classification Report:')
    print(classification_report(y, y_pred))


X_train, X_test, y_train, y_test = prepare_data(churn_data)
ann_model = build_ann()
ann_model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)
evaluate_model(ann_model, X_test, y_test)
