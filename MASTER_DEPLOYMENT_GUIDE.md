# Monte Carlo Simulator: AWS Elastic Beanstalk Master Deployment Guide

This master guide provides complete instructions for deploying your Monte Carlo Simulator to AWS Elastic Beanstalk, consolidating all the individual configuration guides.

## Deployment Package

Your deployment package is located at:
```
C:\Users\henri\Desktop\monte_carlo_simulator\monte_carlo_simulator.zip
```

## Step 1: Initial Setup

1. Log in to the [AWS Management Console](https://console.aws.amazon.com/)
2. Navigate to the Elastic Beanstalk service
3. Click "Create application"

### Application Information

- **Application name**: `monte-carlo-simulator`
- **Application tags**: Optional, can leave empty

## Step 2: Environment Tier

- Choose **Web server environment**
  - This creates an environment for a web application accessible over HTTP

## Step 3: Platform Configuration

- **Platform type**: Select **Managed platform**
- **Platform**: Select **Python**
- **Platform branch**: Select **Python 3.9 running on 64bit Amazon Linux 2**
- **Platform version**: Choose the **recommended/latest version**

## Step 4: Application Code

- Select **Upload your code**
- Click the **Upload** button
- Select the file from your computer:
  - `C:\Users\henri\Desktop\monte_carlo_simulator\monte_carlo_simulator.zip`
- For Version label: Use `v1` or `initial-version`

## Step 5: Presets

- Select **Single instance (free tier eligible)**
  - For better performance but more cost, you can choose a different preset later

## Step 6: Service Access Configuration

> Detailed instructions available in [SERVICE_ACCESS_GUIDE.md](./SERVICE_ACCESS_GUIDE.md)

### Service Role

- Choose **Create and use new service role**
- Default name: `aws-elasticbeanstalk-service-role` is fine

### EC2 Instance Profile

- Choose **Create and use new instance profile**
- Default name: `aws-elasticbeanstalk-ec2-role` is fine

### EC2 Key Pair

- Leave as **None** (unless you need SSH access)

## Step 7: Capacity Configuration

> Detailed instructions available in [CAPACITY_CONFIG_GUIDE.md](./CAPACITY_CONFIG_GUIDE.md)

For development and testing:

- **Environment type**: Single instance
- **Instance type**: t2.medium (recommended for Monte Carlo simulations)
  - You can use t2.micro for free tier, but performance will be limited
- **Capacity**: Min/Max = 1/1 instance

## Step 8: Load Balancer Configuration

- For Single instance: Skip this section (not applicable)
- For Load balanced: Use default settings

## Step 9: Rolling Updates & Deployments Configuration

- Use the default settings:
  - Deployment policy: All at once
  - Deployment preferences: Ignore health check

## Step 10: Security Configuration

- Use default settings
- EC2 key pair: None (unless you specifically need SSH access)

## Step 11: Monitoring

- Use default settings (with CloudWatch monitoring enabled)
- X-Ray daemon: Disabled (unless you need advanced tracing)

## Step 12: Software Configuration

> Detailed instructions available in [SOFTWARE_CONFIG_GUIDE.md](./SOFTWARE_CONFIG_GUIDE.md)

This is the **most critical section** for your Streamlit application:

### Environment Properties

- Add a new environment variable:
  - Name: `PORT`
  - Value: `8501`

### Log Streaming

- Keep all log options enabled (default)

## Step 13: Final Review and Create

1. Review all configuration settings
2. Click "Create environment"
3. Wait for environment creation (5-10 minutes)
4. Monitor the "Events" tab for progress and any errors

## After Deployment

Once your environment shows "OK" status (green):

1. Click the provided URL to access your Monte Carlo Simulator
2. Verify that the application loads correctly
3. Test core functionality:
   - Historical data loading
   - Monte Carlo simulations
   - Visualization of results
   - Optimization calculations

## Troubleshooting Common Issues

### Application Not Loading

- **Health shows "Degraded" or "Severe"**:
  - Check the "Logs" section
  - Look for Python errors or missing dependencies
  - Verify the PORT environment variable is set correctly

- **502 Bad Gateway Error**:
  - Indicates your application failed to start
  - Check if Streamlit is binding to the correct port
  - Review logs for Python exceptions

### Missing Dependencies

- If certain dependencies are missing:
  1. Update your requirements.txt file
  2. Create a new deployment bundle
  3. Upload the new version in the Elastic Beanstalk console

### Performance Issues

- If the application is slow:
  - Consider upgrading to a larger instance type
  - Monitor CPU and memory usage in CloudWatch

## Maintenance and Updates

### Updating Your Application

1. Make changes to your code locally
2. Create a new ZIP file:
   ```
   cd monte_carlo_simulator && git archive -o monte_carlo_simulator_v2.zip HEAD
   ```
3. In the Elastic Beanstalk console:
   - Go to your environment
   - Click "Upload and deploy"
   - Upload the new ZIP file
   - Specify a new version label (e.g., "v2")
   - Click "Deploy"

### Monitoring Costs

- Monitor your AWS Billing dashboard
- Consider terminating the environment when not in use:
  - In the Elastic Beanstalk console
  - Select your environment
  - Click "Actions" > "Terminate environment"

## Additional Resources

- [AWS Elastic Beanstalk Documentation](https://docs.aws.amazon.com/elasticbeanstalk/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy/)
- [AWS Elastic Beanstalk Pricing](https://aws.amazon.com/elasticbeanstalk/pricing/)
