# Run the PlantCareNet website locally

From this project folder, open Terminal and run:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501). Open it in your browser, upload a JPG, PNG, or WEBP leaf photo, and view the prediction. The app uses `model.keras` and `class_names.json` from this folder.
