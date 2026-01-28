# Deployment Guide

This guide covers deploying the BBB Permeability Prediction GUI to Streamlit Cloud and other platforms.

## Streamlit Cloud Deployment

### Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at https://streamlit.io/cloud)
3. Model artifacts (see `ARTIFACTS_GUIDE.md`)

### Step-by-Step Deployment

#### 1. Prepare Your Repository

1. **Create a new GitHub repository** (or use existing)
2. **Upload all files** from the `BBB FINAL GUI` folder:
   - `streamlit_app.py` (required)
   - `requirements.txt` (required)
   - `README.md` (recommended)
   - `ARTIFACTS_GUIDE.md` (optional)
   - `DEPLOYMENT.md` (optional)
   - `.gitignore` (recommended)

#### 2. Add Model Artifacts

**Option A: Add artifacts to repository (recommended for small files)**
- Create `artifacts/` directory in your repository
- Upload all model files:
  ```
  artifacts/
  ├── descriptor_cols.json
  ├── stage2_feature_cols.json
  └── models/
      ├── stage1_efflux.joblib
      ├── stage1_influx.joblib
      ├── stage1_pampa.joblib
      ├── stage1_cns.joblib
      └── stage2_bbb.joblib
  ```

**Option B: Use Git LFS for large files**
- If artifacts are too large for regular Git, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.joblib"
  git add .gitattributes
  git add artifacts/
  git commit -m "Add model artifacts"
  ```

**Option C: Upload via Streamlit Cloud (if supported)**
- Some platforms allow file uploads after deployment

#### 3. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Click "New app"**
3. **Connect your GitHub repository**
4. **Configure deployment**:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.8 or higher
5. **Click "Deploy"**

#### 4. Verify Deployment

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Run `streamlit_app.py`
  - Provide a public URL

- **Check the app**:
  - Navigate to the provided URL
  - Test file upload functionality
  - Verify descriptor computation works
  - If artifacts are present, test predictions

### Troubleshooting

#### Common Issues

1. **"Module not found" errors**
   - Check `requirements.txt` includes all dependencies
   - Ensure RDKit is listed (may need `rdkit-pypi` on some platforms)

2. **"Artifacts not found"**
   - Verify `artifacts/` directory is in repository root
   - Check file paths match exactly
   - Ensure files are committed (not in `.gitignore`)

3. **"Import error: libXrender.so.1"**
   - This is expected on Streamlit Cloud
   - Molecular visualization will be disabled
   - All other functionality works normally

4. **App won't start**
   - Check Streamlit Cloud logs
   - Verify `streamlit_app.py` has no syntax errors
   - Ensure Python version is compatible

## Local Deployment

### Development Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

### Production Server

For production deployment, consider:
- Docker containerization
- Reverse proxy (nginx)
- Process manager (systemd, supervisor)

## File Size Considerations

- **Model artifacts** can be large (several MB each)
- **GitHub free tier**: 100 MB file size limit
- **Git LFS**: Recommended for files > 50 MB
- **Alternative storage**: Consider cloud storage (S3, Google Drive) with download on startup

## Environment Variables

If needed, you can set environment variables in Streamlit Cloud:
- Go to app settings
- Add secrets/environment variables
- Access in code: `os.environ.get('VARIABLE_NAME')`

## Updating the App

1. Make changes to `streamlit_app.py`
2. Commit and push to GitHub
3. Streamlit Cloud automatically redeploys
4. Changes go live within minutes

## Support

For deployment issues:
- Check Streamlit Cloud documentation: https://docs.streamlit.io/streamlit-community-cloud
- Review app logs in Streamlit Cloud dashboard
- Contact: Dr. Sivanesan Dakshanamurthy - sd233@georgetown.edu
