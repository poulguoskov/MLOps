import requests
import streamlit as st

# Title
st.set_page_config(page_title="Clickbait Detector", page_icon="🙅🏼‍♀️")
st.title("🙅🏼‍♀️ Clickbait Classifier")
st.write("Type a headline below to check if it's clickbait or not!")

# Cloud Run URL
API_URL = "https://clickbait-api-gcp-136485552734.europe-west1.run.app/predict"

# Input field
text_input = st.text_area("Enter headline here:", height=100)

# Button
if st.button("Check if it's clickbait 🚀"):
    if text_input:
        with st.spinner("Analyzing...🧐"):
            try:
                # Send text to your API
                payload = {"text": text_input}
                response = requests.post(API_URL, params=payload)

                if response.status_code == 200:
                    result = response.json()
                    is_clickbait = result["is_clickbait"]

                    # Show result
                    if is_clickbait:
                        st.error("🚨 WARNING! This is CLICKBAIT! 🚨")
                        st.image("https://media.giphy.com/media/3ornk6UHtk276vLtkY/giphy.gif", width=300)
                    else:
                        st.success("✅ This seems safe (Not clickbait).")
                else:
                    st.error(f"Something went wrong with the API: {response.status_code}")

            except Exception as e:
                st.error(f"Could not connect to the server: {e}")
    else:
        st.warning("You must enter something first!")
