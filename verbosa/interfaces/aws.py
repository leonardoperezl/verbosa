from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import boto3


@dataclass(frozen=True)
class AWSCredentials:
    _instances: ClassVar[
        Dict[Tuple[str, str, str, Optional[str]], "AWSCredentials"]
    ] = {}
    
    access_key_id: str
    secret_access_key: str
    region_name: str
    session_token: Optional[str] = None
    
    def __new__(
        cls,
        access_key_id: str,
        secret_access_key: str,
        region_name: str,
        session_token: Optional[str] = None
    ) -> AWSCredentials:
        key = (access_key_id, secret_access_key, region_name, session_token)
        
        # If we've already created this combination, reuse it
        if key in cls._instances:
            return cls._instances[key]
        
        # Otherwise, create a new instance and store it
        self = super().__new__(cls)
        cls._instances[key] = self
        return self
    
    def to_boto3_session(self) -> boto3.Session:
        return boto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
            aws_session_token=self.session_token
        )


@dataclass
class AthenaDataBaseDetails:
    database: str
    workgroup: Optional[str] = "primary"
    s3_output_location: Optional[str] = None
    ctas_approach: bool = False
    ctas_parameters: Optional[Mapping[str, str | Sequence[str]]] = None
    unload_approach: bool = False
    unload_parameters: Optional[Mapping[str, str | Sequence[str]]] = None