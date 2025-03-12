-- Drop existing table if it exists with old primary key
DROP TABLE IF EXISTS price_history;

CREATE TABLE IF NOT EXISTS price_history (
    datetime TIMESTAMP WITH TIME ZONE,
    symbol VARCHAR(10),
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume BIGINT,
    PRIMARY KEY (datetime, symbol)
);

-- Create index on symbol alone (still useful for symbol-only queries)
CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history(symbol);
