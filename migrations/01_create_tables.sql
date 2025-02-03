-- Drop views first
DROP VIEW IF EXISTS recent_predictions CASCADE;
DROP VIEW IF EXISTS prediction_performance CASCADE;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS technical_indicators CASCADE;

-- Create technical_indicators table first (as it's referenced by predictions)
CREATE TABLE technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    price DECIMAL NOT NULL,
    macd_color TEXT,
    rsi DECIMAL,
    volume_trend TEXT,
    macd_value DECIMAL,
    macd_signal DECIMAL,
    macd_histogram DECIMAL,
    short_term_direction TEXT,
    volatility TEXT,
    trend_strength_adx TEXT,
    short_term_momentum DECIMAL,
    short_term_acceleration DECIMAL,
    short_term_velocity DECIMAL,
    trend_strength DECIMAL,
    atr DECIMAL,
    bb_upper DECIMAL,
    bb_middle DECIMAL,
    bb_lower DECIMAL,
    stoch_k DECIMAL,
    stoch_d DECIMAL,
    adx DECIMAL
);

-- Create predictions table
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    prediction TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence INTEGER,
    price_at_prediction DECIMAL NOT NULL,
    actual_direction TEXT,
    verified BOOLEAN DEFAULT FALSE,
    correct BOOLEAN,
    technical_indicator_id BIGINT REFERENCES technical_indicators(id)
);

-- Create performance_metrics table
CREATE TABLE performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    total_predictions INTEGER,
    correct_predictions INTEGER,
    accuracy DECIMAL,
    avg_confidence DECIMAL,
    best_streak INTEGER,
    current_streak INTEGER
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_verified ON predictions(verified);
CREATE INDEX idx_technical_indicators_timestamp ON technical_indicators(timestamp);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);

-- Create view for recent predictions with technical indicators
CREATE VIEW recent_predictions AS
SELECT 
    p.*,
    ti.price,
    ti.macd_color,
    ti.rsi,
    ti.volume_trend,
    ti.macd_value,
    ti.macd_signal,
    ti.macd_histogram,
    ti.short_term_direction,
    ti.volatility,
    ti.trend_strength_adx
FROM predictions p
JOIN technical_indicators ti ON p.technical_indicator_id = ti.id
WHERE p.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY p.timestamp DESC;

-- Create view for prediction performance analysis
CREATE VIEW prediction_performance AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(AVG(confidence)::numeric, 2) as avg_confidence,
    ROUND((SUM(CASE WHEN correct THEN 1 ELSE 0 END)::float / COUNT(*) * 100)::numeric, 2) as accuracy
FROM predictions
WHERE verified = true
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Add comments for documentation
COMMENT ON TABLE technical_indicators IS 'Stores market analysis data and technical indicators';
COMMENT ON TABLE predictions IS 'Stores 30-second price movement predictions and their verification';
COMMENT ON TABLE performance_metrics IS 'Tracks overall prediction accuracy and statistics';
COMMENT ON VIEW recent_predictions IS 'Shows recent predictions with their technical indicators';
COMMENT ON VIEW prediction_performance IS 'Hourly performance analysis of predictions';