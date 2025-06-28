# AWS Elastic Beanstalk Capacity Configuration Guide

After configuring Service Access, you'll likely encounter the Capacity configuration section. Here's how to configure it optimally for your Monte Carlo Simulator.

## Capacity Configuration

### Environment Type

You'll need to choose between two environment types:

1. **Single instance environment** (Recommended for initial setup)
   - Deploy to a single EC2 instance
   - No load balancer (reduced cost)
   - No auto-scaling
   - Perfect for development, testing, and low-traffic applications
   - **Free tier eligible** if you choose the t2.micro instance type

2. **Load balanced environment**
   - Deploy to multiple EC2 instances
   - Includes a load balancer (additional cost)
   - Supports auto-scaling
   - Provides high availability
   - Good for production applications with significant traffic

### Instance Types

For a **Single instance environment**:

- For free tier (recommended for testing):
  - **t2.micro** (1 vCPU, 1 GiB RAM)
  - Free for 750 hours per month for the first 12 months

- For better performance (recommended for our Monte Carlo Simulator):
  - **t2.medium** (2 vCPU, 4 GiB RAM)
  - Better choice for compute-intensive tasks
  - Not free tier eligible, but reasonably priced

### Capacity Settings

For a **Single instance environment**:

- Keep the defaults:
  - Min: 1 instance
  - Max: 1 instance

For a **Load balanced environment** (if selected):

- Instance settings:
  - Min: 1 instance
  - Max: 2 instances (adjust based on expected load)
  - Scale up/down thresholds: Keep defaults initially

### Spot Instance Requests (Optional)

- For cost savings, you can enable spot instance requests
- Set a maximum price you're willing to pay
- Be aware that spot instances can be terminated if demand increases
- For initial setup, you can leave this disabled

## Recommended Configuration for Monte Carlo Simulator

For development and testing:

- **Environment type**: Single instance
- **Instance type**: t2.medium (the t2.micro might be too underpowered for the mathematical computations)
- **Capacity**: Min/Max = 1/1 instance

For production (if needed later):

- **Environment type**: Load balanced
- **Instance type**: t2.medium or larger
- **Capacity**: Min/Max = 1/3 instances (adjust based on traffic)
- **Scaling trigger**: CPU utilization > 70% for 5 minutes

## After Capacity Configuration

After completing the capacity configuration:

1. Click **Next** to continue to the next section
2. You'll likely configure:
   - Load balancer (if using a load balanced environment)
   - Updates and deployments
   - Monitoring
   - Managed updates

## Cost Considerations

- **Single instance with t2.micro**: Lowest cost, free tier eligible
- **Single instance with t2.medium**: Better performance, moderate cost
- **Load balanced environment**: Higher cost but better availability

Remember that you can always modify these settings later if your requirements change.
