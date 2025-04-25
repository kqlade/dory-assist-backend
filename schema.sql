-- ① Reminders ───────────────────────────────────────────────────────────
CREATE TABLE reminders (
    reminder_id     UUID        PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    reminder_text   TEXT        NOT NULL,
    channel         TEXT        NOT NULL DEFAULT 'sms',
    status          TEXT        NOT NULL DEFAULT 'pending',   -- pending | processing | sent | failed
    next_fire_at    TIMESTAMPTZ,                              -- nullable for location/event triggers
    last_error      TEXT,
    payload         JSONB       NOT NULL,                     -- full ReminderTask inc. triggers
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX reminders_due_idx     ON reminders (status, next_fire_at);
CREATE INDEX reminders_user_idx    ON reminders (user_id, created_at DESC);

-- ② Message envelopes ───────────────────────────────────────────────────
CREATE TABLE message_envelopes (
    envelope_id UUID PRIMARY KEY,
    user_id     TEXT,
    channel     TEXT      CHECK (channel IN ('sms', 'mms', 'email')),
    instruction TEXT      NOT NULL,          -- first human-readable line
    payload     JSONB,                       -- full envelope inc. images
    status      TEXT      NOT NULL DEFAULT 'received',  -- received | awaiting_user | processed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX envelopes_user_idx    ON message_envelopes (user_id, created_at DESC);
CREATE INDEX envelopes_status_idx  ON message_envelopes (status);
