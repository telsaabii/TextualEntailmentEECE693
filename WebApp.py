import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, BertTokenizer

# Import the model architecture
from model_architecture import ExplainableNLISystem

# Set page configuration
st.set_page_config(
    page_title="Explainable Natural Language Inference",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    model = ExplainableNLISystem()
    
    # Load from saved checkpoints
    try:
        model.explainer.model = T5ForConditionalGeneration.from_pretrained("./explanation_generator/final")
        model.explainer.tokenizer = T5Tokenizer.from_pretrained("./explanation_generator/final")
        
        model.predictor.model = BertForSequenceClassification.from_pretrained("./label_predictor/final")
        model.predictor.tokenizer = BertTokenizer.from_pretrained("./label_predictor/final")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure the model has been trained and saved correctly.")
    
    return model

def main():
    # Title and description
    st.title("Explainable Natural Language Inference")
    st.markdown("""
    This application demonstrates explainable natural language inference using a two-stage model:
    1. **Explanation Generator**: Generates an explanation for the relationship between premise and hypothesis
    2. **Label Predictor**: Predicts the label (entailment, contradiction, neutral) based on the explanation
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    model_type = st.sidebar.radio(
        "Select Model Domain",
        ["General (e-SNLI)", "Philosophy"]
    )
    
    # Load the appropriate model
    if model_type == "General (e-SNLI)":
        model_path = "./explanation_generator/final"  # Replace with your actual path
    else:
        model_path = "./philosophy_nli/explainer/final"  # Replace with your actual path
    
    # Load the model
    model = load_model()
    
    # Main content area - tabs
    tab1, tab2, tab3 = st.tabs(["Interactive Demo", "Batch Prediction", "Model Evaluation"])
    
    # Tab 1: Interactive Demo
    with tab1:
        st.header("Interactive Demo")
        
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            premise = st.text_area("Premise", height=100, placeholder="Enter a premise statement here...")
        
        with col2:
            hypothesis = st.text_area("Hypothesis", height=100, placeholder="Enter a hypothesis statement here...")
        
        # Prediction button
        if st.button("Generate Explanation & Predict"):
            if premise and hypothesis:
                with st.spinner("Generating explanation and prediction..."):
                    # Make prediction
                    try:
                        result = model.predict(premise, hypothesis)
                        
                        # Display results
                        st.subheader("Results")
                        
                        # Use color-coding for the label
                        label = result['label']
                        if label == "entailment":
                            label_color = "green"
                        elif label == "contradiction":
                            label_color = "red"
                        else:
                            label_color = "blue"
                        
                        st.markdown(f"**Predicted Label:** <span style='color:{label_color};font-size:1.2em;'>{label.upper()}</span>", unsafe_allow_html=True)
                        
                        st.markdown("**Generated Explanation:**")
                        st.info(result['explanation'])
                        
                        # Visualization of the process
                        st.subheader("Model Process Visualization")
                        st.image("https://via.placeholder.com/800x200?text=Two-Stage+Model+Process+Visualization", caption="Illustration of the two-stage prediction process")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please enter both premise and hypothesis.")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file with premise and hypothesis columns", type="csv")
        
        if uploaded_file is not None:
            # Load the data
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check if required columns exist
                required_cols = ["premise", "hypothesis"]
                if all(col in df.columns for col in required_cols):
                    st.success("File loaded successfully!")
                    st.dataframe(df.head())
                    
                    # Process button
                    if st.button("Process Batch"):
                        with st.spinner("Processing batch data..."):
                            # Create empty lists for results
                            explanations = []
                            labels = []
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            
                            # Process each row
                            for i, row in df.iterrows():
                                # Update progress
                                progress_bar.progress((i + 1) / len(df))
                                
                                # Make prediction
                                result = model.predict(row['premise'], row['hypothesis'])
                                
                                # Store results
                                explanations.append(result['explanation'])
                                labels.append(result['label'])
                            
                            # Add results to dataframe
                            df['explanation'] = explanations
                            df['predicted_label'] = labels
                            
                            # Display results
                            st.subheader("Results")
                            st.dataframe(df)
                            
                            # Label distribution
                            st.subheader("Label Distribution")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            label_counts = df['predicted_label'].value_counts()
                            sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
                            ax.set_title("Distribution of Predicted Labels")
                            ax.set_ylabel("Count")
                            ax.set_xlabel("Label")
                            st.pyplot(fig)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                else:
                    st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Tab 3: Model Evaluation
    with tab3:
        st.header("Model Evaluation")
        
        # Load and display evaluation metrics
        try:
            # Placeholder for model evaluation metrics
            st.subheader("Explanation Quality Metrics")
            
            eval_data = {
                "BLEU": 0.42,
                "ROUGE-1": 0.65,
                "ROUGE-2": 0.48,
                "ROUGE-L": 0.62
            }
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BLEU", f"{eval_data['BLEU']:.2f}")
            col2.metric("ROUGE-1", f"{eval_data['ROUGE-1']:.2f}")
            col3.metric("ROUGE-2", f"{eval_data['ROUGE-2']:.2f}")
            col4.metric("ROUGE-L", f"{eval_data['ROUGE-L']:.2f}")
            
            # Classification metrics
            st.subheader("Classification Metrics")
            
            # Placeholder confusion matrix
            conf_matrix = np.array([
                [850, 120, 30],
                [90, 780, 130],
                [40, 110, 850]
            ])
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                yticklabels=['Entailment', 'Neutral', 'Contradiction'],
                ax=ax
            )
            ax.set_title("Confusion Matrix")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            
            # Placeholder classification report
            report_data = {
                "Class": ["Entailment", "Neutral", "Contradiction", "Accuracy", "Macro Avg", "Weighted Avg"],
                "Precision": [0.87, 0.83, 0.89, None, 0.86, 0.86],
                "Recall": [0.85, 0.78, 0.85, None, 0.83, 0.83],
                "F1-Score": [0.86, 0.80, 0.87, 0.83, 0.84, 0.84],
                "Support": [1000, 1000, 1000, 3000, 3000, 3000]
            }
            
            # Convert to dataframe and display
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df)
            
        except Exception as e:
            st.error(f"Error loading evaluation metrics: {e}")
            st.info("Please make sure the model has been evaluated and metrics are saved.")

if __name__ == "__main__":
    main()