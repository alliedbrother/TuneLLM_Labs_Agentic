"""AWS cloud GPU provider service.

Lists existing EC2 GPU instances. AWS instances must be launched via the AWS console
or CLI — this provider detects and manages existing ones.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# GPU instance types and their VRAM
AWS_GPU_INSTANCES = {
    "p3.2xlarge": {"gpu": "V100", "gpu_count": 1, "vram_gb": 16, "vcpus": 8, "ram_gb": 61},
    "p3.8xlarge": {"gpu": "V100", "gpu_count": 4, "vram_gb": 16, "vcpus": 32, "ram_gb": 244},
    "p4d.24xlarge": {"gpu": "A100", "gpu_count": 8, "vram_gb": 40, "vcpus": 96, "ram_gb": 1152},
    "g4dn.xlarge": {"gpu": "T4", "gpu_count": 1, "vram_gb": 16, "vcpus": 4, "ram_gb": 16},
    "g4dn.2xlarge": {"gpu": "T4", "gpu_count": 1, "vram_gb": 16, "vcpus": 8, "ram_gb": 32},
    "g5.xlarge": {"gpu": "A10G", "gpu_count": 1, "vram_gb": 24, "vcpus": 4, "ram_gb": 16},
    "g5.2xlarge": {"gpu": "A10G", "gpu_count": 1, "vram_gb": 24, "vcpus": 8, "ram_gb": 32},
    "g6.xlarge": {"gpu": "L4", "gpu_count": 1, "vram_gb": 24, "vcpus": 4, "ram_gb": 16},
}


class AWSProvider:
    """List and manage AWS EC2 GPU instances.

    Uses AWS access key + secret key for authentication.
    Instances should be launched via AWS console/CLI — this provider
    detects existing running GPU instances.
    """

    def __init__(self, access_key_id: str, secret_access_key: str, region: str = "us-east-1"):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region

    async def list_instances(self) -> list[dict]:
        """List running EC2 GPU instances.

        Uses the EC2 DescribeInstances API with boto3.
        """
        try:
            import boto3

            ec2 = boto3.client(
                "ec2",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region,
            )

            # Filter for running GPU instances
            gpu_types = list(AWS_GPU_INSTANCES.keys())
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "instance-state-name", "Values": ["running"]},
                    {"Name": "instance-type", "Values": gpu_types},
                ]
            )

            instances = []
            for reservation in response.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    inst_type = inst.get("InstanceType", "")
                    gpu_info = AWS_GPU_INSTANCES.get(inst_type, {})
                    name_tag = ""
                    for tag in inst.get("Tags", []):
                        if tag.get("Key") == "Name":
                            name_tag = tag.get("Value", "")

                    instances.append({
                        "id": inst.get("InstanceId", ""),
                        "name": name_tag or inst.get("InstanceId", ""),
                        "status": inst.get("State", {}).get("Name", "unknown"),
                        "instance_type": inst_type,
                        "gpu_type": gpu_info.get("gpu", "Unknown"),
                        "gpu_count": gpu_info.get("gpu_count", 0),
                        "gpu_ram_gb": gpu_info.get("vram_gb", 0),
                        "ip_address": inst.get("PublicIpAddress"),
                        "private_ip": inst.get("PrivateIpAddress"),
                        "region": self.region,
                    })
            return instances

        except ImportError:
            logger.warning("boto3 not installed — AWS provider unavailable")
            return []
        except Exception as e:
            logger.error(f"Failed to list AWS instances: {e}")
            raise

    async def destroy_instance(self, instance_id: str) -> bool:
        """Terminate an EC2 instance."""
        try:
            import boto3

            ec2 = boto3.client(
                "ec2",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region,
            )
            ec2.terminate_instances(InstanceIds=[instance_id])
            return True
        except Exception as e:
            logger.error(f"Failed to terminate AWS instance {instance_id}: {e}")
            return False
