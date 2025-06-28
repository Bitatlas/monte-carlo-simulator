# AWS Elastic Beanstalk Form Guide

Here's how to fill out each section of the AWS Elastic Beanstalk creation form:

## 1. Platform Info

- **Platform type**: Select **Managed platform**
  - This is the standard option where AWS maintains the platform

- **Platform**: Select **Python**
  - This matches our Streamlit application which is built with Python

- **Platform branch**: Select **Python 3.9 running on 64bit Amazon Linux 2**
  - This version matches what we've configured in our application

- **Platform version**: Choose the **recommended/latest version**
  - The default selection is fine

## 2. Application Code

- Select **Upload your code**
- Click the **Upload** button
- Select the file from your computer:
  - `C:\Users\henri\Desktop\monte_carlo_simulator\monte_carlo_simulator.zip`
- For Version label: Use `v1` or `initial-version`

## 3. Presets

- Select **Single instance (free tier eligible)**
  - This is the most cost-effective option for testing
  - You won't be charged for the first 750 hours of usage per month

## 4. Additional Configuration

After clicking "Create application," you'll be taken to a configuration page. Make these changes:

### Software Configuration
- Click on "Configuration" from the left sidebar
- Find "Software" and click "Edit"
- Add environment property:
  - Name: `PORT`
  - Value: `8501`
- Click "Apply"

## 5. Create Application

- Once all settings are configured, click the "Create application" button
- The deployment process will begin and may take 5-10 minutes
- You can monitor the progress in the Events tab

## 6. Accessing Your Application

- Once deployment is complete, you'll see a URL at the top of the page
- Click this URL to access your Monte Carlo Simulator
