
import shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tempfile
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the pre-trained model
lstm_model = pickle.load(open('lstm_model.pkl', 'rb'))


# Constants
sequence_length = 50
# Update the expected features according to the number of columns in your txt file (after dropping unwanted columns)
expected_num_features = 16
maintenance_threshold = 0.9 # Threshold for deciding if maintenance is needed

# Function to preprocess input data
def preprocess_input(raw_data, sequence_length):
    # Drop empty columns if any (NaNs)
    raw_data.dropna(axis=1, inplace=True)

    # Assign column names as per your feature set
    raw_data.columns = [
        'id', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
        's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
        's18', 's19', 's20', 's21'
    ]

    # Debug: Print the columns after assignment
    # st.write("Columns after assignment:", raw_data.columns.tolist())
    # st.write("Data shape after column assignment:", raw_data.shape{selected_id})
    st.write("Data shape for selected Engine ID: ", raw_data.shape)

    # Drop columns that are constant or less important
    raw_data = raw_data.drop(columns=['setting3', 's1', 's5', 's10', 's16', 's18', 's19', 's14', 's6'])

    # Debug: Print the columns after dropping
    # st.write("Columns after dropping:", raw_data.columns.tolist())
    # st.write("Data shape after dropping columns:", raw_data.shape)

    # Calculate RUL
    raw_data.sort_values(['id', 'cycle'], inplace=True)
    rul = pd.DataFrame(raw_data.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    raw_data = raw_data.merge(rul, on=['id'], how='left')
    raw_data['RUL'] = raw_data['max'] - raw_data['cycle']
    raw_data.drop('max', axis=1, inplace=True)

    # Filter the relevant columns
    raw_data['cycle_norm'] = raw_data['cycle']
    cols_normalize = raw_data.columns.difference(['id', 'cycle', 'RUL'])

    # Normalize the data
    scaler = MinMaxScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(raw_data[cols_normalize]),
                             columns=cols_normalize,
                             index=raw_data.index)

    # Debug: Print the normalized columns
    # st.write("Normalized columns:", norm_data.columns.tolist())

    # Rejoin with the non-normalized columns
    final_df = raw_data[['id', 'cycle', 'RUL']].join(norm_data)
    raw_data = final_df.reindex(columns=final_df.columns)

    # Debug: Final columns check
    # st.write("Final DataFrame columns:", raw_data.columns.tolist())
    # st.write("Final DataFrame shape:", raw_data.shape)

    # Debugging Step: Check the number of columns after normalization and joining
    if raw_data.shape[1] != expected_num_features + 3:  # +3 for id, cycle, and RUL
        st.error(f"Expected {expected_num_features + 3} columns, but got {raw_data.shape[1]}")
        st.write("Current columns:", raw_data.columns.tolist())
        return None, None

    # Step 8: Ensure all required columns are used to generate sequences
    all_feature_columns = norm_data.columns.tolist()

    # Check if there are enough data points to create sequences
    if len(raw_data) < sequence_length:
        st.warning("Not enough sequence data, choose another engine ID.")
        return None, None

    # Generate sequences
    sequences = []

    for start in range(len(raw_data) - sequence_length + 1):
        end = start + sequence_length
        sequences.append(raw_data.iloc[start:end][all_feature_columns].values)

    sequences = np.array(sequences)

    # Check if the sequences match the required shape
    if sequences.shape[1:] != (sequence_length, expected_num_features):
        st.error(f"Expected shape (n_samples, {sequence_length}, {expected_num_features}), but got {sequences.shape}")
        return None, None

    return sequences, all_feature_columns

# New Function: Classify the last sequence
def classify_last_sequence(processed_data):
    last_sequence = processed_data[-1].reshape(1, sequence_length, expected_num_features)
    last_prediction = lstm_model.predict(last_sequence)
    classification = int(last_prediction > maintenance_threshold)
    return classification

def generate_recommendations(predictions, threshold=0.9):
    binary_classification = (predictions > threshold).astype(int)

    # Check for five consecutive '1's in the binary classification
    maintenance_needed_cycles = []
    consecutive_count = 0
    first_cycle = None

    for i in range(len(binary_classification)):
        if binary_classification[i] == 1:
            consecutive_count += 1
            if consecutive_count == 1:
                first_cycle = i + sequence_length   # first cycle = 50 seq
        else:
            consecutive_count = 0  # Reset the count if '1' is not consecutive

        if consecutive_count == 5:
            maintenance_needed_cycles.append(first_cycle)
            consecutive_count = 0  # Reset after detecting the sequence

    if maintenance_needed_cycles:
        recommendation_message = f"Maintenance should have required starting from cycle: {maintenance_needed_cycles [0]}"
    else:
        recommendation_message = "Machine last sequence (1) maintenance required in current cycle (0) operating normally, no maintenance required"

    return recommendation_message, binary_classification


# Streamlit App
def run_pdm_dashboard():
    st.title("Predictive Maintenance (PdM) Dashboard")

    uploaded_file = st.file_uploader("Choose a TXT file", type="txt")

    if uploaded_file is not None:
        # Read the uploaded file
        raw_data = pd.read_csv(uploaded_file, sep=" ", header=None)
        st.write("Uploaded Data:")
        st.write(raw_data)

        # Select engine ID for analysis
        selected_id = st.selectbox("Select Engine ID", options=range(1, 21))

        # Filter data based on selected ID
        filtered_data = raw_data[raw_data.iloc[:, 0] == selected_id]

        if filtered_data.empty:
            st.warning(f"No data available for Engine ID {selected_id}. Please select a different ID.")
            return

        st.write(f"Filtered Data for Engine ID {selected_id}:")
        st.write(filtered_data)

        # Process the data
        processed_data, all_feature_columns = preprocess_input(filtered_data, sequence_length)

        if processed_data is not None:
            # st.write("Processed Data Shape: ", processed_data.shape)
            st.write("Processed Data Shape: ", processed_data[0].shape if processed_data is not None else "No data")

            # Make predictions
            predictions = lstm_model.predict(processed_data)
            if predictions is None:
                st.error("Predictions are None.")
                return
            recommendation_message, binary_classification = generate_recommendations(predictions, threshold=maintenance_threshold)

            # Classify the last sequence
            last_sequence_classification = classify_last_sequence(processed_data)
            st.write(f"Last Sequence Classification: {last_sequence_classification}")
            st.write("*(1 = Failure Occurs)  *(0 = No Failure Occurs)")
            st.write(" ")

            # Starting cycle from sequence_length
            start_cycle = sequence_length
            end_cycle = start_cycle + len(predictions.flatten()) - 1

            result_df = pd.DataFrame({
                'Cycle': range(start_cycle, end_cycle + 1),
                'Prediction': predictions.flatten(),
                'Failure_Class': binary_classification.flatten()
            })

            st.write("Predictions with Maintenance Decisions:")
            st.write(result_df)

            # Plotting the graph with cycle on x-axis and failure prediction on y-axis
            # st.line_chart(result_df.set_index('Cycle')['Prediction'])
            # st.write("x-axis: Cycles, y-axis: Failure Prediction")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(result_df['Cycle'], result_df['Prediction'])

            # Adding labels and title
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Failure Prediction')
            ax.set_title('Failure Prediction Over Cycles')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Provide recommendations
            st.markdown("**Recommendations:**")
            st.write(recommendation_message)

            # SHAP Force Plot
            shap.initjs()
            explainer = shap.GradientExplainer(lstm_model, processed_data)
            shap_values = explainer.shap_values(processed_data)

            # Use SHAP force plot to explain the last sequence prediction
            i = -1  # Last sequence index
            # Compute the baseline output for the last sequence
            baseline_input = np.expand_dims(processed_data[i], axis=0)
            base_value = lstm_model.predict(baseline_input)[0][0]

            last_timestep = processed_data[i][-1].reshape(1,-1) # Select only the last timestep

            # Ensure you're only working with the last timestep (1 row with 16 features)
            last_sequence_shap_values = shap_values[0][i].reshape(16)
            last_sequence_data_df = pd.DataFrame(last_timestep, columns=all_feature_columns)  # Only pass 16 columns for the last timestep



            explanation_last = shap.Explanation(values=last_sequence_shap_values,
                                                base_values=base_value,
                                                data=last_sequence_data_df)
            st.write("\n")
            st.markdown("**SHAP Force Plot for Last Sequence Prediction:**")

            # Create a temporary file to save the SHAP plot
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                shap.force_plot(explanation_last, matplotlib=True)
                plt.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
                st.image(tmpfile.name)
            st.write("Force Plot >> individual predictions, how each feature contributes to a prediction")
            st.markdown("**Red bars:**")
            st.write("  - Increased the prediction (pushing it towards a higher probability of failure)")
            st.markdown("**Blue bars:**")
            st.write("  - Decreased the prediction (pushing it towards a lower probability of failure)")
            st.write("\n")
            st.write("\n")

            # Summarize SHAP values across time steps to aggregate feature importance
            shap_values_selected = shap_values[0][:, :, 0]  # (time steps, features)
            shap_values_aggregated = np.sum(shap_values_selected, axis=0)  # Sum across time steps
            # SHAP Summary Plot
            st.markdown("**SHAP Summary Plot (Overall Sequence):**")
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(all_feature_columns, shap_values_aggregated, color=["deeppink" if val > 0 else "dodgerblue" for val in shap_values_aggregated])

            # Add SHAP values on the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.5f}',
                        va='center', ha='center', color='black')

            ax.set_xlabel('SHAP value (impact on model output)')
            ax.set_ylabel('Feature')
            ax.set_title(f'SHAP summary for Engine ID {selected_id}')
            plt.tight_layout()
            st.pyplot(fig)

            st.write("Summary Plot >> overall importance of various features across all time steps")
            st.markdown("**Blue bars:**")
            st.write("  - Features that reduce the model's output (lower failure probability)")
            st.markdown("**Pink bars:**")
            st.write("  - Features that increase the model's output (higher failure probability)")


if __name__ == "__main__":
    run_pdm_dashboard()
