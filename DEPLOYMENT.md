# CalcBB Deployment Guide

## Streamlit Cloud

1. Push the **GitHub Submission** folder contents to a new GitHub repository (root = this folder).
2. On [share.streamlit.io](https://share.streamlit.io), connect the repository.
3. Configure:
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.9 or 3.10
4. Deploy. The app will use the `artifacts/` directory in the repo.

## Important

- Ensure all files in `artifacts/` are committed (including `stage2_modelC/*.pkl`).
- If artifacts exceed GitHub's 100MB limit, consider Git LFS.
