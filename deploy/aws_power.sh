#!/usr/bin/env bash
# Switch the HealthGraph AWS deployment on/off.
#   bash deploy/aws_power.sh status   # state + current URL
#   bash deploy/aws_power.sh stop     # stop the instance (no compute charges while stopped)
#   bash deploy/aws_power.sh start    # start it again; prints the NEW URL (public IP changes)
# The container restarts by itself on boot (docker --restart unless-stopped); the app is
# reachable about 1-2 minutes after the instance reports "running".
set -euo pipefail
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
IID=$(aws ec2 describe-instances --filters Name=tag:Project,Values=healthgraph \
      Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query 'Reservations[0].Instances[0].InstanceId' --output text)
[[ "$IID" == "None" || -z "$IID" ]] && { echo "No HealthGraph instance found."; exit 1; }

show() {
  read -r STATE IP < <(aws ec2 describe-instances --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text)
  echo "instance $IID: $STATE"
  [[ "$IP" != "None" ]] && echo "UI:   http://$IP/ui" && echo "Docs: http://$IP/docs"
}

case "${1:-status}" in
  status) show ;;
  stop)   aws ec2 stop-instances --instance-ids "$IID" >/dev/null
          aws ec2 wait instance-stopped --instance-ids "$IID"; show ;;
  start)  aws ec2 start-instances --instance-ids "$IID" >/dev/null
          aws ec2 wait instance-running --instance-ids "$IID"; show
          IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
          echo -n "waiting for the app"; for _ in $(seq 1 40); do
            curl -sf -m 5 "http://$IP/health" >/dev/null 2>&1 && { echo " - ready."; exit 0; }; echo -n "."; sleep 6; done
          echo " - not responding yet; try again in a minute." ;;
  *)      echo "usage: $0 {status|start|stop}"; exit 1 ;;
esac
