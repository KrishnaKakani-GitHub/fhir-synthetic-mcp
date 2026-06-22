"""Pydantic v2 data models for the Clinical AI Governance Platform."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


class Patient(BaseModel):
    id: str
    name: str
    birth_date: date
    gender: Gender
    mrn: str


class Observation(BaseModel):
    id: str
    patient_id: str
    code: str
    display: str
    value: float
    unit: str
    effective_date: date


class DataStore(BaseModel):
    """Root model for the legacy JSON fixture (seed only)."""
    patients: list[Patient] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)


class ProposedObservation(BaseModel):
    patient_id: str
    code: str
    display: str
    value: float
    unit: str
    effective_date: date


class PendingWriteStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class PendingWrite(BaseModel):
    write_id: str
    resource_type: str = "Observation"
    proposed: ProposedObservation
    status: PendingWriteStatus = PendingWriteStatus.pending
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    # Clinical warnings from the deterministic validator (non-blocking)
    validation_warnings: list[str] = Field(default_factory=list)
