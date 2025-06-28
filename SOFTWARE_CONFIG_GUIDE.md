# AWS Elastic Beanstalk Software Configuration Guide

The Software configuration section is critical for your Monte Carlo Simulator, particularly for ensuring the Streamlit application runs correctly.

## Software Configuration

### Platform Software

This section defines the software stack for your environment:

- For most Python applications, keep the default settings
- The proxy server is usually set to Nginx by default, which works fine for Streamlit

### Environment Properties

This is the **most critical section** for your Streamlit application. You must add:

1. Required environment variable:
   - Name: `PORT`
   - Value: `8501`
   - This ensures Streamlit binds to the correct port that Elastic Beanstalk expects

2. Optional additional variables (if needed):
   - `STREAMLIT_SERVER_ENABLE_CORS`: `false`
   - `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`: `false`
   - `PYTHONPATH`: `/var/app` (if you encounter import errors)

### Environment Configuration

1. **Log streaming**: Keep enabled for easy access to logs

2. **Instance log streaming to CloudWatch Logs**: 
   - Recommended to keep enabled
   - Helps with debugging and monitoring

3. **Log group settings**:
   - Keep the default settings
   - Retention: 7 days (default) is usually sufficient

### CloudWatch integration (if available)

- Instance metrics: Keep enabled
- Health reporting system metrics: Keep enabled
- These help monitor your application's performance

## Why the PORT Environment Variable is Critical

- Elastic Beanstalk routes traffic to your application on a specific port
- By setting `PORT=8501`, you ensure Streamlit binds to this port
- Our Procfile (already created) references this environment variable:
  ```
  web: streamlit run app.py --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false --server.address=0.0.0.0
  ```
- Without this configuration, Elastic Beanstalk won't be able to reach your application

## After Software Configuration

After completing the software configuration:

1. Review all settings once more
2. Click "Create environment" or "Apply" 
3. Wait for the environment creation process to complete (5-10 minutes)
4. Monitor the "Events" log for any errors
5. Once the environment shows "OK" status, your Monte Carlo Simulator should be accessible via the provided URL

## Troubleshooting Common Software Issues

If your application fails to start:

1. Check environment logs for errors:
   - "Request failed with status code 502" often means your application didn't start correctly
   - Look for Python errors in the logs

2. Common issues:
   - Missing PORT environment variable
   - Missing dependencies in requirements.txt
   - Application errors preventing startup

3. Solutions:
   - Verify the PORT environment variable is set to 8501
   - Check the application logs for specific error messages
   - If needed, update your application and deploy a new version
