# AWS Elastic Beanstalk Deployment Guide

This guide will help you deploy the Monte Carlo Simulator to AWS Elastic Beanstalk.

## Prerequisites

1. An AWS account
2. AWS CLI installed and configured with your credentials
3. AWS Elastic Beanstalk CLI (already installed)
4. Git repository cloned locally

## Steps to Deploy

### 1. Initialize Elastic Beanstalk

Navigate to your project directory and run:

```bash
cd monte_carlo_simulator
eb init
```

When prompted:
- Select your AWS region (typically the one closest to you)
- Enter your AWS credentials if asked
- Create a new application named "monte-carlo-simulator"
- Select "Python" as the platform
- Choose Python 3.9 as the platform version
- Select "No" when asked if you want to use CodeCommit
- Choose "Yes" when asked if you want to set up SSH

### 2. Create an Elastic Beanstalk Environment

```bash
eb create monte-carlo-env
```

This will create a new environment named "monte-carlo-env". The process may take 5-10 minutes as AWS sets up all necessary resources.

### 3. Monitor the Deployment

You can monitor the deployment status with:

```bash
eb status
```

And view detailed events with:

```bash
eb events
```

### 4. View the Application Logs (if needed)

If you encounter issues, you can view the logs with:

```bash
eb logs
```

### 5. Open the Application

Once the deployment is complete, open your application in a web browser:

```bash
eb open
```

## Managing Your Deployment

### Updating Your Application

After making changes to your code:

1. Commit your changes to Git
2. Deploy the updated application:

```bash
eb deploy
```

### Terminating the Environment

When you're done using the application, you can terminate the environment to avoid incurring charges:

```bash
eb terminate monte-carlo-env
```

### Scaling Your Application

If you need to handle more traffic:

1. Go to the AWS Management Console
2. Navigate to the Elastic Beanstalk service
3. Select your environment
4. Click on "Configuration" in the left sidebar
5. Under "Capacity", click "Edit"
6. Adjust the number of instances or instance type

## Troubleshooting

### Common Issues

1. **Deployment Timeout**
   - Check the health of your environment with `eb health`
   - View logs with `eb logs`
   - Increase the deployment timeout in configuration

2. **Health Check Failures**
   - Ensure your application is binding to the correct port (8501)
   - Check if Streamlit is configured correctly in the Procfile
   - Verify that all dependencies are installed

3. **Package Installation Errors**
   - Check for any system dependencies in the error logs
   - Add missing dependencies to the .ebextensions config file

### Getting Support

If you encounter issues, you can:
- Check AWS Elastic Beanstalk documentation
- View the Elastic Beanstalk section of the AWS Management Console
- Check CloudWatch Logs for detailed application logs

## Cost Management

By default, Elastic Beanstalk uses a t2.medium instance. To optimize costs:

1. Go to the AWS Management Console
2. Navigate to the Elastic Beanstalk service
3. Select your environment
4. Click on "Configuration" in the left sidebar
5. Under "Capacity", click "Edit"
6. Change instance type to t2.micro for the lowest cost (if your application can run on it)
7. Consider enabling "Environment time-based scaling" to scale down during non-business hours

## Security Considerations

1. Set up HTTPS with AWS Certificate Manager
2. Configure proper security groups
3. Use IAM roles with least privilege
4. Enable AWS WAF if needed for additional security

Remember to terminate your environment when not in use to avoid unnecessary charges.
