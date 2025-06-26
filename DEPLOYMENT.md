# Deployment Guide

This guide explains how to deploy the Monte Carlo Simulator to Firebase using Cloud Run.

## Prerequisites

1. [GitHub Account](https://github.com)
2. [Google Cloud Account](https://cloud.google.com)
3. [Firebase Account](https://firebase.google.com)
4. [Git](https://git-scm.com/downloads)
5. [Firebase CLI](https://firebase.google.com/docs/cli)
6. [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
7. [Docker](https://www.docker.com/products/docker-desktop/) (for local testing)

## Initial Setup

### 1. Create a GitHub Repository

1. Go to GitHub and create a new repository named `monte-carlo-simulator`
2. Push your code to the repository:

```bash
# Add all files to git
git add .

# Commit the files
git commit -m "Initial commit"

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/monte-carlo-simulator.git

# Push to GitHub
git push -u origin main
```

### 2. Create a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Click "Add Project"
3. Name your project "monte-carlo-simulator"
4. Follow the setup wizard (enable Google Analytics if desired)
5. Once created, note your Project ID for later use

### 3. Set Up Google Cloud Project

The Firebase project automatically creates a Google Cloud project with the same ID.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (it should match your Firebase project ID)
3. Enable the following APIs:
   - Cloud Run API
   - Container Registry API
   - Cloud Build API

### 4. Create Service Account for GitHub Actions

1. In the Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Name: "github-actions"
4. Description: "Service account for GitHub Actions CI/CD"
5. Grant the following roles:
   - Cloud Run Admin
   - Storage Admin
   - Service Account User
   - Cloud Build Editor
6. Click "Create Key" and download the JSON key file

## Configuring GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your repository > Settings > Secrets and variables > Actions
2. Add the following secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: The entire content of the service account JSON key file
   - `FIREBASE_PROJECT_ID`: Your Firebase project ID
   - `FIREBASE_SERVICE_ACCOUNT`: Generate this using the Firebase CLI

To generate the Firebase service account token:

```bash
firebase login:ci
```

## Deployment Options

### Option 1: Manual Deployment

#### Deploy to Cloud Run directly

```bash
# Build the Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/monte-carlo-simulator .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/monte-carlo-simulator

# Deploy to Cloud Run
gcloud run deploy monte-carlo-simulator \
  --image gcr.io/YOUR_PROJECT_ID/monte-carlo-simulator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Deploy to Firebase Hosting

```bash
# Initialize Firebase (if not already done)
firebase init hosting

# Deploy to Firebase
firebase deploy --only hosting
```

### Option 2: Automated Deployment with GitHub Actions

Simply push your code to the `main` branch, and the GitHub Actions workflow will automatically:

1. Build the Docker image
2. Push it to Google Container Registry
3. Deploy to Cloud Run
4. Update Firebase Hosting

## Testing Your Deployment

After deployment, you can access your application at:

- Cloud Run URL: https://monte-carlo-simulator-[unique-id].a.run.app
- Firebase URL: https://monte-carlo-simulator.web.app

## Troubleshooting

### Common Issues

1. **"Error: Container failed to start"**
   - Check Cloud Run logs to see what's happening
   - Ensure Streamlit port is set to 8080 in the Dockerfile

2. **"Firebase Hosting rewrite failed"**
   - Verify that the Cloud Run service is deployed and public
   - Check service name in firebase.json matches the deployed service

3. **"Permission denied" in GitHub Actions**
   - Ensure the service account has all required permissions
   - Verify the GCP_SA_KEY secret is properly formatted

### Viewing Logs

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=monte-carlo-simulator" --limit=50

# View Firebase Hosting logs
firebase hosting:log
```

## Costs and Scaling

- Cloud Run charges only for the time your containers are running and processing requests
- Set memory limits and CPU allocation based on your application's needs
- Firebase Hosting has a generous free tier, but check pricing for high-traffic sites

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firebase Hosting Documentation](https://firebase.google.com/docs/hosting)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy/docker)
