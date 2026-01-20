import time

import requests
import streamlit as st

# Title
st.set_page_config(page_title="Clickbait Detector", page_icon="🙅🏼‍♀️")
st.title("🙅🏼‍♀️ Clickbait Classifier")
st.write("Type a headline below to check if it's clickbait or not!")

# Backend options
BACKENDS = {
    "PyTorch (GCP)": {
        "url": "https://clickbait-api-gcp-136485552734.europe-west1.run.app/classify",
        "method": "json",
    },
    "ONNX (Lightweight)": {
        "url": "https://clickbait-api-onnx-136485552734.europe-west1.run.app/classify",
        "method": "json",
    },
    "BentoML (Adaptive Batching)": {
        "url": "https://clickbait-bentoml-136485552734.europe-west1.run.app/classify",
        "method": "json_list",
    },
}

# Sidebar settings
st.sidebar.header("Settings")
selected_backend = st.sidebar.selectbox("Select Backend:", list(BACKENDS.keys()))
show_details = st.sidebar.checkbox("Show technical details", value=False)

if show_details:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Backend Info:**")
    st.sidebar.text(f"URL: {BACKENDS[selected_backend]['url'][:40]}...")

# Input field
text_input = st.text_area("Enter headline here:", height=100)

# Button
if st.button("Check if it's clickbait 🚀"):
    if text_input:
        with st.spinner("Analyzing...🧐"):
            try:
                backend = BACKENDS[selected_backend]
                start_time = time.time()

                # Send request based on backend type
                if backend["method"] == "json":
                    response = requests.post(backend["url"], json={"text": text_input}, timeout=30)
                elif backend["method"] == "json_list":
                    response = requests.post(backend["url"], json={"texts": [text_input]}, timeout=30)

                elapsed_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()

                    # Handle different response formats
                    if isinstance(result, list):
                        result = result[0]  # BentoML returns list

                    is_clickbait = result["is_clickbait"]
                    confidence = result.get("confidence", None)

                    # Show result
                    if is_clickbait:
                        st.error("🚨 WARNING! This is CLICKBAIT! 🚨")
                        st.image("https://media.giphy.com/media/3ornk6UHtk276vLtkY/giphy.gif", width=300)
                    else:
                        st.success("✅ This seems safe (Not clickbait).")

                    # Show confidence and response time
                    col1, col2 = st.columns(2)
                    if confidence:
                        col1.metric("Confidence", f"{confidence:.1%}")
                    col2.metric("Response Time", f"{elapsed_time:.2f}s")

                elif response.status_code == 403:
                    st.error("🔒 API requires authentication. Try a different backend.")
                else:
                    st.error(f"Something went wrong with the API: {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The backend might be cold-starting.")
            except Exception as e:
                st.error(f"Could not connect to the server: {e}")
    else:
        st.warning("You must enter something first!")

# Batch mode section
st.markdown("---")
with st.expander("📦 Batch Mode - Classify multiple headlines"):
    batch_input = st.text_area(
        "Enter multiple headlines (one per line):",
        height=150,
        placeholder="Headline 1\nHeadline 2\nHeadline 3",
        key="batch_input",
    )

    if st.button("Classify All", key="batch_button"):
        if batch_input.strip():
            headlines = [h.strip() for h in batch_input.strip().split("\n") if h.strip()]

            if len(headlines) > 0:
                with st.spinner(f"Classifying {len(headlines)} headlines..."):
                    backend = BACKENDS[selected_backend]
                    start_time = time.time()

                    try:
                        if backend["method"] == "json_list":
                            # BentoML supports native batch
                            response = requests.post(backend["url"], json={"texts": headlines}, timeout=60)
                            if response.status_code == 200:
                                results = response.json()
                        else:
                            # Other backends: send one by one
                            results = []
                            for headline in headlines:
                                resp = requests.post(backend["url"], json={"text": headline}, timeout=30)
                                if resp.status_code == 200:
                                    results.append(resp.json())

                        elapsed_time = time.time() - start_time

                        # Display results
                        st.metric("Total Time", f"{elapsed_time:.2f}s")

                        for i, result in enumerate(results):
                            is_clickbait = result["is_clickbait"]
                            confidence = result.get("confidence", 0)
                            icon = "🚨" if is_clickbait else "✅"
                            label = "CLICKBAIT" if is_clickbait else "Safe"
                            headline_display = headlines[i][:50] + ("..." if len(headlines[i]) > 50 else "")
                            st.write(f"{icon} **{headline_display}** → {label} ({confidence:.1%})")

                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Enter at least one headline.")
