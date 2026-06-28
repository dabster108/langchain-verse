CREATE TABLE IF NOT EXISTS customers (
    account_id TEXT PRIMARY KEY,
    phone TEXT,
    email TEXT,
    is_flagged_for_review BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    txn_id TEXT UNIQUE,
    user_id TEXT NOT NULL,
    account_id TEXT REFERENCES customers(account_id),
    amount NUMERIC(14, 2) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'NPR',
    merchant_id TEXT,
    device_id TEXT,
    ip_address TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_risk_results (
    id BIGSERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    agent_name TEXT NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence_score DOUBLE PRECISION NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_risk_results_transaction_id
    ON agent_risk_results(transaction_id);

CREATE TABLE IF NOT EXISTS otp_sessions (
    otp_session_id VARCHAR PRIMARY KEY,
    txn_id VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    channel_1_otp VARCHAR,
    channel_1_sent_at TIMESTAMP,
    channel_1_verified_at TIMESTAMP,
    channel_1_status VARCHAR DEFAULT 'PENDING',
    channel_2_otp VARCHAR,
    channel_2_sent_at TIMESTAMP,
    channel_2_verified_at TIMESTAMP,
    channel_2_status VARCHAR DEFAULT 'PENDING',
    final_decision VARCHAR,
    sim_swap_suspected BOOLEAN DEFAULT FALSE,
    attempt_count_ch1 INT DEFAULT 0,
    attempt_count_ch2 INT DEFAULT 0,
    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id),
    FOREIGN KEY (account_id) REFERENCES customers(account_id)
);

CREATE TABLE IF NOT EXISTS account_flags (
    flag_id SERIAL PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    flag_type VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    reason VARCHAR,
    FOREIGN KEY (account_id) REFERENCES customers(account_id)
);
