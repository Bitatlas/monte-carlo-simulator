# AWS Management Console Deployment Guide

Since there are issues with the EB CLI, this guide shows how to deploy the Monte Carlo Simulator using the AWS Management Console.

## Prerequisites

1. An AWS account
2. Your code pushed to GitHub (already done)
3. The Monte Carlo Simulator codebase with AWS configuration files (already prepared)

## Step 1: Create an Application Bundle

1. Create a ZIP file of your application:
   - Make sure to include all files and directories
   - Do not include the parent directory in the ZIP
   - Do not include `.git` directory or other unnecessary files

You can use this command to create the ZIP file (run from the monte_carlo_simulator directory):

```
git archive -o monte_carlo_simulator.zip HEAD
```

## Step 2: Create a New Elastic Beanstalk Application

1. Log in to the [AWS Management Console](https://console.aws.amazon.com/)
2. Navigate to the Elastic Beanstalk service
3. Click "Create application"
4. Enter these details:
   - Application name: `monte-carlo-simulator`
   - Platform: Python
   - Platform branch: Python 3.9
   - Platform version: The latest recommended version
   - Application code: Upload your code
     - Choose "Upload your code"
     - Upload the ZIP file you created
5. Click "Create application"

## Step 3: Configure Environment Settings

After clicking "Create application", you'll be taken to the configuration page:

1. Under "Service access":
   - Create and use a new service role
   - Create and use a new EC2 instance profile

2. Under "Capacity":
   - Environment type: Single instance (for cost savings)
   - Instance type: t2.medium (or t2.micro for minimum cost)

3. Under "Load balancer" (if shown):
   - Keep the default settings

4. Under "Rolling updates and deployments":
   - Deployment policy: All at once (for simplicity)

5. Under "Security":
   - EC2 key pair: No selection (unless you want SSH access)

6. Under "Monitoring":
   - Keep default settings

7. Under "Managed updates":
   - Keep default settings

8. Click "Save" at the bottom

## Step 4: Configure Additional Options

1. Click on "Configuration" from the environment overview page
2. Edit the "Software" configuration:
   - Set environment property: `PORT` = `8501`
   - Click "Apply"

## Step 5: Monitor Deployment

1. Once you've submitted the application, you'll be taken to the environment dashboard
2. Monitor the "Events" tab to see the deployment progress
3. Deployment typically takes 5-10 minutes

## Step 6: Access Your Application

1. Once the environment status shows "OK" (green), click on the environment URL
2. This will open your Monte Carlo Simulator in your browser

## Troubleshooting

If your application doesn't deploy correctly:

1. Check the "Events" tab for error messages
2. View the application logs:
   - Go to "Logs" in the left navigation panel
   - Request logs or view last 100 lines
3. Common issues:
   - Ensure your Procfile has the correct command format
   - Check that all dependencies are in requirements.txt
   - Verify your application is configured to run on the port specified in environment variables

## Updating Your Application

To update your application after making changes:

1. Create a new ZIP file with your updated code
2. In the Elastic Beanstalk console, go to your environment
3. Click "Upload and deploy" on the environment overview page
4. Upload your new ZIP file
5. Click "Deploy"

## Cost Management

Monitor your costs:

1. Set up billing alerts in the AWS Billing dashboard
2. Consider terminating the environment when not in use
3. To terminate:
   - Go to your environment in the Elastic Beanstalk console
   - Click "Actions" > "Terminate environment"
   - Confirm termination
