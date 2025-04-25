from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Index

Base = declarative_base()

class Reminder(Base):
    __tablename__ = "reminders"
    reminder_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    reminder_text = Column(Text, nullable=False)
    reminder_time = Column(DateTime(timezone=True))
    timezone = Column(String)
    channel = Column(String, nullable=False, default="sms")
    status = Column(String, nullable=False, default="pending")
    last_error = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime(timezone=True))
    payload = Column(JSON)

class MessageEnvelope(Base):
    __tablename__ = "message_envelopes"
    envelope_id = Column(String, primary_key=True)
    user_id = Column(String)
    channel = Column(String)
    instruction = Column(Text, nullable=False)
    payload = Column(JSON)
    raw_refs = Column(JSON)
    status = Column(String, nullable=False, default="received")
    created_at = Column(DateTime)
    __table_args__ = (
        Index("ix_message_envelopes_status", "status"),
    ) 