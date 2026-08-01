import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Neural Network From Scratch",layout="wide")
st.title("Neural NetwoSrk From Scratch")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

tab1, tab2 = st.tabs(["Training","Prediction"])

with tab1:
    uploaded_file = st.file_uploader("Upload CSV Dataset",type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select Target Column",df.columns)

        X = df.drop(columns=[target_column]).values.astype(np.float64)
        y = df[target_column].values.reshape(-1,1).astype(np.float64)

        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)

        st.subheader("Model Configuration")
        hidden_neurons = st.slider("Hidden Layer Neurons",2,64,16)

        learning_rate = st.slider("Learning Rate",0.0001,1.0,0.01)

        epochs = st.slider("Epochs",100,20000,5000,step=100)

        if st.button("Train Neural Network"):
            np.random.seed(42)
            input_neurons = X_normalized.shape[1]
            output_neurons = 1

            weights_input_hidden = np.random.randn(input_neurons,hidden_neurons) * np.sqrt(1 / input_neurons)

            bias_hidden = np.zeros((1, hidden_neurons))

            weights_hidden_output = np.random.randn(hidden_neurons,output_neurons) * np.sqrt(1 / hidden_neurons)

            bias_output = np.zeros((1, output_neurons))

            def relu(x):
                return np.maximum(0, x)

            def relu_derivative(x):
                return (x > 0).astype(float)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            def binary_cross_entropy(y_true,y_pred):
                epsilon = 1e-8
                y_pred = np.clip(y_pred,epsilon,1 - epsilon)
                return -np.mean(y_true * np.log(y_pred)+(1 - y_true) * np.log(1 - y_pred))

            progress_bar = st.progress(0)
            loss_placeholder = st.empty()

            for epoch in range(epochs):

                hidden_input = np.dot(X_normalized,weights_input_hidden) + bias_hidden
                hidden_output = relu(hidden_input)

                final_input = np.dot(hidden_output,weights_hidden_output) + bias_output

                predicted_output = sigmoid(final_input)
                loss = binary_cross_entropy(y,predicted_output)
                output_error = (predicted_output - y)

                d_weights_hidden_output = np.dot(hidden_output.T,output_error) / len(X_normalized)

                d_bias_output = np.mean(output_error,axis=0,keepdims=True)

                hidden_error = np.dot(output_error,weights_hidden_output.T) * relu_derivative(hidden_input)

                d_weights_input_hidden = np.dot(X_normalized.T,hidden_error) / len(X_normalized)

                d_bias_hidden = np.mean(hidden_error,axis=0,keepdims=True)

                weights_hidden_output -= (learning_rate *d_weights_hidden_output)

                bias_output -= (learning_rate *d_bias_output)

                weights_input_hidden -= (learning_rate *d_weights_input_hidden)

                bias_hidden -= (learning_rate *d_bias_hidden)

                if epoch % 100 == 0:
                    progress = min(epoch / epochs,1.0)
                    progress_bar.progress(progress)
                    loss_placeholder.info(f"Epoch: {epoch} | Loss: {loss:.6f}")

            predictions = (predicted_output > 0.5).astype(int)

            accuracy = np.mean(predictions == y) * 100

            st.success("Training Completed Successfully")

            st.metric("Accuracy",f"{accuracy:.2f}%")

            result_df = pd.DataFrame({
                "Actual":y.flatten(),
                "Predicted Probability":predicted_output.flatten(),
                "Predicted Class":predictions.flatten()})

            st.subheader("Prediction Results")
            st.dataframe(result_df)
            st.session_state.model_trained = True
            st.session_state.weights_input_hidden = weights_input_hidden
            st.session_state.bias_hidden = bias_hidden

            st.session_state.weights_hidden_output = weights_hidden_output
            st.session_state.bias_output = bias_output

            st.session_state.X_min = X_min
            st.session_state.X_max = X_max

            st.session_state.feature_columns = df.drop(columns=[target_column]).columns

with tab2:
    st.subheader("Custom Prediction")
    if st.session_state.model_trained:
        custom_inputs = []
        for col in st.session_state.feature_columns:
            value = st.number_input(f"Enter {col}",value=0.0,key=col)
            custom_inputs.append(value)

        if st.button("Predict"):
            def relu(x):
                return np.maximum(0, x)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            custom_data = np.array([custom_inputs]).astype(np.float64)
            custom_data = (custom_data -st.session_state.X_min) / (st.session_state.X_max -st.session_state.X_min +1e-8)

            hidden_input = np.dot(custom_data,st.session_state.weights_input_hidden) + st.session_state.bias_hidden
            hidden_output = relu(hidden_input)

            final_input = np.dot(hidden_output,st.session_state.weights_hidden_output) + st.session_state.bias_output
            prediction = sigmoid(final_input)

            predicted_class = (prediction > 0.5).astype(int)
            st.subheader("Prediction Output")
            st.write(f"Prediction Probability: {prediction[0][0]:.4f}")
            st.write(f"Predicted Class: {predicted_class[0][0]}")

    else:
        st.warning("Please train the model first in the Training tab.")