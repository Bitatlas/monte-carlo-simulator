# AWS Elastic Beanstalk Service Access Configuration Guide

This guide helps you configure the service access settings for your Elastic Beanstalk environment.

## Service Access Configuration

### Service Role

A service role allows AWS Elastic Beanstalk to manage resources on your behalf.

1. In the "Service role" dropdown:
   - Choose **Create and use new service role**
   - Default name: `aws-elasticbeanstalk-service-role` (can be kept as is)
   - This role will automatically have the required policies attached

2. If you've used Elastic Beanstalk before and have existing roles:
   - You can select an existing role with proper permissions
   - Look for roles that include "elasticbeanstalk" in their names

### EC2 Instance Profile

An EC2 instance profile allows your EC2 instances to interact with other AWS services.

1. In the "EC2 instance profile" dropdown:
   - Choose **Create and use new instance profile**
   - Default name: `aws-elasticbeanstalk-ec2-role` (can be kept as is)
   - This profile will automatically have the required policies attached

2. If you have existing instance profiles:
   - You can select an existing profile with proper permissions
   - Look for profiles that include "elasticbeanstalk" in their names

### EC2 Key Pair (Optional)

An EC2 key pair allows you to SSH into your EC2 instances for troubleshooting.

1. For initial setup:
   - You can leave this **blank** if you don't need SSH access
   - This is recommended for simplicity

2. If you want SSH access:
   - Choose an existing key pair from the dropdown
   - Or click "Create new key pair" to create one
   - **Important**: You need to download and securely store the private key when creating a new key pair. It cannot be downloaded again later.

## Recommended Settings for Beginners

For the simplest and most secure setup:

- **Service role**: Create and use new service role
- **EC2 instance profile**: Create and use new instance profile
- **EC2 key pair**: Leave as "None" unless you specifically need SSH access

These settings will allow AWS to create and manage all necessary permissions automatically.

## Notes on IAM Permissions

If you're creating new roles and seeing "Permission Denied" errors:

1. Make sure your AWS user has the necessary IAM permissions:
   - `iam:CreateRole`
   - `iam:AttachRolePolicy`
   - `iam:CreateInstanceProfile`
   - `iam:AddRoleToInstanceProfile`

2. If you're part of an organization, you might need to ask your AWS administrator to:
   - Create these roles for you, or
   - Grant you the necessary permissions to create roles

## After Configuration

After completing the service access configuration:

1. Click **Save** or **Next** to continue
2. Review other environment configuration settings
3. Complete the application creation process
