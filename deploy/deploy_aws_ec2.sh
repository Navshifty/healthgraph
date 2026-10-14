#!/usr/bin/env bash
# Deploy the HealthGraph Docker app to a free-tier AWS EC2 instance (t3.micro, Amazon Linux 2023).
#
# Prerequisite (once, yourself): `aws configure` with an IAM user that can use EC2 and S3.
# Usage:   bash deploy/deploy_aws_ec2.sh [region]          (default ap-south-1)
# Cleanup: bash deploy/deploy_aws_ec2.sh --destroy [region]
#
# What it creates (all tagged Project=healthgraph):
#   * a private S3 bucket holding the ~5 MB build context (code + model bundle)
#   * a security group that opens only TCP 80
#   * one t3.micro with a 12 GB gp3 disk; cloud-init adds 2 GB swap, installs Docker,
#     downloads the context via a 24 h presigned URL, builds the image and runs it on port 80.
set -euo pipefail
cd "$(dirname "$0")/.."
DESTROY=0; [[ "${1:-}" == "--destroy" ]] && { DESTROY=1; shift; }
REGION="${1:-ap-south-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="healthgraph-deploy-${ACCOUNT}"
export AWS_DEFAULT_REGION="$REGION"

if [[ $DESTROY == 1 ]]; then
  IIDS=$(aws ec2 describe-instances --filters Name=tag:Project,Values=healthgraph \
         Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
  # Release the Elastic IP first: an unattached Elastic IP is billed by the hour.
  for A in $(aws ec2 describe-addresses --filters Name=tag:Project,Values=healthgraph --query 'Addresses[].AllocationId' --output text); do
    ASSOC=$(aws ec2 describe-addresses --allocation-ids "$A" --query 'Addresses[0].AssociationId' --output text)
    [[ "$ASSOC" != "None" ]] && aws ec2 disassociate-address --association-id "$ASSOC"
    aws ec2 release-address --allocation-id "$A"
  done
  [[ -n "$IIDS" ]] && aws ec2 terminate-instances --instance-ids $IIDS >/dev/null && aws ec2 wait instance-terminated --instance-ids $IIDS
  SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=healthgraph-web --query 'SecurityGroups[0].GroupId' --output text)
  [[ "$SG" != "None" ]] && aws ec2 delete-security-group --group-id "$SG"
  aws s3 rb "s3://$BUCKET" --force || true
  echo "HealthGraph AWS resources removed."; exit 0
fi

STAGE=$(mktemp -d)
cp -r Dockerfile .dockerignore requirements-serve.txt healthgraph app "$STAGE"/
mkdir -p "$STAGE/artifacts" && cp artifacts/healthgat_uci.pt "$STAGE/artifacts/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
tar --force-local -czf "$STAGE.tar.gz" -C "$STAGE" .

aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || {
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true; }
aws s3 cp "$STAGE.tar.gz" "s3://$BUCKET/healthgraph_ctx.tar.gz" --only-show-errors
URL=$(aws s3 presign "s3://$BUCKET/healthgraph_ctx.tar.gz" --expires-in 86400)

VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=healthgraph-web --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --group-name healthgraph-web --description "HealthGraph API/UI: HTTP only" \
       --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null
fi
AMI=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*-kernel-6.*-x86_64" \
      "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

cat > "$STAGE/userdata.sh" <<EOF
#!/bin/bash
set -euxo pipefail
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
dnf install -y docker
systemctl enable --now docker
mkdir -p /opt/healthgraph && cd /opt/healthgraph
curl -fsSL "$URL" -o ctx.tar.gz
tar xzf ctx.tar.gz && rm ctx.tar.gz
docker build -t healthgraph:latest .
docker run -d --name healthgraph --restart unless-stopped -p 80:7860 healthgraph:latest
docker image prune -f
echo "HEALTHGRAPH_DEPLOYED"
EOF

IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type t3.micro --security-group-ids "$SG" \
  --user-data "file://$STAGE/userdata.sh" --metadata-options HttpTokens=required \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":12,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=healthgraph},{Key=Project,Value=healthgraph}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance $IID running at $IP. The image builds on first boot (~10-15 min); then open http://$IP/ui"
