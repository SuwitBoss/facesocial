-- Antispoofing Service Tables
-- Handles liveness detection and spoof detection to prevent face spoofing attacks

-- Antispoofing checks - การตรวจสอบ antispoofing
CREATE TABLE antispoofing.antispoofing_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64) NOT NULL,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Check details
    check_type VARCHAR(20) NOT NULL, -- 'passive', 'active_liveness', 'motion_based'
    input_type VARCHAR(20) NOT NULL, -- 'image', 'video', 'live_stream'
    input_hash VARCHAR(64) NOT NULL,
    input_metadata JSONB, -- Image/video metadata
    
    -- Face region information
    face_bbox JSONB, -- Bounding box of detected face
    face_quality_score DECIMAL(5,4),
    face_landmarks JSONB, -- Facial landmarks for analysis
    
    -- Active liveness details (if applicable)
    liveness_challenges JSONB, -- ["turn_left", "turn_right", "blink", "nod", "smile"]
    challenge_responses JSONB, -- Response data for each challenge
    challenge_completion_rate DECIMAL(5,4), -- Percentage of challenges completed
    
    -- Passive liveness analysis
    texture_analysis JSONB, -- Texture-based liveness features
    color_analysis JSONB, -- Color space analysis
    frequency_analysis JSONB, -- Frequency domain analysis
    
    -- Motion-based analysis (for video)
    motion_vectors JSONB, -- Motion analysis data
    temporal_consistency JSONB, -- Temporal consistency checks
    micro_movements JSONB, -- Micro-movement detection
    
    -- Results
    is_live BOOLEAN,
    confidence_score DECIMAL(5,4),
    liveness_score DECIMAL(5,4), -- Overall liveness score
    spoof_type VARCHAR(30), -- 'photo', 'video', 'mask', 'screen', 'deepfake', null if live
    spoof_confidence DECIMAL(5,4), -- Confidence that this is a specific spoof type
    risk_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    
    -- Detailed analysis results
    analysis_details JSONB, -- Detailed analysis breakdown
    feature_scores JSONB, -- Individual feature scores
    model_predictions JSONB, -- Predictions from different models
    
    -- Processing details
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    models_used JSONB, -- Array of models used in analysis
    
    -- Quality metrics
    image_quality_score DECIMAL(5,4),
    lighting_quality DECIMAL(5,4),
    pose_quality DECIMAL(5,4),
    
    -- Context
    request_source VARCHAR(50), -- 'login', 'payment', 'verification', 'registration'
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Security context
    session_risk_score DECIMAL(5,4), -- Risk assessment for this session
    device_fingerprint VARCHAR(128), -- Device fingerprint hash
    location_info JSONB, -- Geographic location if available
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Liveness sessions - เซสชัน liveness check แบบ active
CREATE TABLE antispoofing.liveness_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(128) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Session configuration
    challenges_required JSONB NOT NULL, -- List of required challenges
    challenge_order JSONB, -- Order of challenges to present
    max_attempts INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 30,
    
    -- Session progress
    current_challenge VARCHAR(20),
    current_challenge_index INTEGER DEFAULT 0,
    challenges_completed JSONB DEFAULT '[]',
    challenges_failed JSONB DEFAULT '[]',
    attempts_count INTEGER DEFAULT 0,
    
    -- Challenge timings
    challenge_start_times JSONB, -- Start time for each challenge
    challenge_completion_times JSONB, -- Completion time for each challenge
    challenge_durations JSONB, -- Duration for each challenge
    
    -- Results
    session_status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'failed', 'expired', 'abandoned'
    overall_result BOOLEAN,
    overall_confidence DECIMAL(5,4),
    failure_reason VARCHAR(100),
    
    -- Session metrics
    total_processing_time INTEGER, -- Total time spent on all challenges
    user_interaction_quality DECIMAL(5,4), -- How well user followed instructions
    session_difficulty VARCHAR(20) DEFAULT 'standard', -- 'easy', 'standard', 'hard'
    
    -- Security
    max_failed_attempts INTEGER DEFAULT 3,
    lockout_duration INTEGER DEFAULT 300, -- 5 minutes
    fraud_indicators JSONB,
    
    -- Device and context
    device_capabilities JSONB, -- Camera capabilities, sensors available
    environment_info JSONB, -- Lighting, background noise, etc.
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Challenge responses - การตอบสนองต่อ challenges
CREATE TABLE antispoofing.challenge_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES antispoofing.liveness_sessions(id),
    challenge_type VARCHAR(30) NOT NULL, -- 'blink', 'turn_left', 'turn_right', 'nod', 'smile', 'speak'
    challenge_instruction TEXT, -- Instruction given to user
    
    -- Response data
    response_data JSONB, -- Raw response data (images, video frames, etc.)
    response_hash VARCHAR(64), -- Hash of response data
    response_quality DECIMAL(5,4), -- Quality of user's response
    
    -- Analysis results
    challenge_success BOOLEAN,
    confidence_score DECIMAL(5,4),
    analysis_details JSONB, -- Detailed analysis of response
    
    -- Timing
    challenge_presented_at TIMESTAMP WITH TIME ZONE,
    response_started_at TIMESTAMP WITH TIME ZONE,
    response_completed_at TIMESTAMP WITH TIME ZONE,
    response_duration_ms INTEGER,
    
    -- Detection details
    movement_detected BOOLEAN,
    movement_quality DECIMAL(5,4),
    facial_features_tracked JSONB, -- Which features were successfully tracked
    
    -- Validation
    spoof_indicators JSONB, -- Any spoof indicators detected
    validation_passed BOOLEAN,
    validation_details JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Spoof detection models - โมเดลสำหรับตรวจจับการปลอมแปลง
CREATE TABLE antispoofing.spoof_detection_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'passive_liveness', 'active_liveness', 'texture_analysis'
    
    -- Model file information
    model_path TEXT NOT NULL,
    model_size_mb DECIMAL(10,2),
    
    -- Model capabilities
    supported_input_types JSONB, -- ['image', 'video', 'live_stream']
    supported_spoof_types JSONB, -- ['photo', 'video', 'mask', 'screen']
    detection_methods JSONB, -- ['texture', 'color', 'motion', 'depth']
    
    -- Performance metrics
    accuracy_score DECIMAL(5,4),
    false_positive_rate DECIMAL(5,4),
    false_negative_rate DECIMAL(5,4),
    processing_speed_fps DECIMAL(8,2),
    
    -- Configuration
    input_requirements JSONB, -- Input format, resolution requirements
    preprocessing_config JSONB,
    threshold_config JSONB, -- Different thresholds for different scenarios
    
    -- Hardware requirements
    min_gpu_memory_mb INTEGER,
    cpu_optimization BOOLEAN DEFAULT FALSE,
    
    -- Status and metadata
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    is_default BOOLEAN DEFAULT FALSE,
    training_dataset TEXT,
    evaluation_dataset TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Device profiles - โปรไฟล์อุปกรณ์สำหรับการวิเคราะห์
CREATE TABLE antispoofing.device_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_fingerprint VARCHAR(128) NOT NULL UNIQUE,
    
    -- Device information
    device_type VARCHAR(30), -- 'mobile', 'tablet', 'desktop', 'webcam'
    device_model VARCHAR(100),
    operating_system VARCHAR(50),
    browser_info VARCHAR(200),
    
    -- Camera capabilities
    camera_resolution JSONB, -- Available resolutions
    camera_features JSONB, -- Features like autofocus, flash, etc.
    video_capabilities JSONB, -- Video recording capabilities
    
    -- Sensors available
    accelerometer_available BOOLEAN DEFAULT FALSE,
    gyroscope_available BOOLEAN DEFAULT FALSE,
    magnetometer_available BOOLEAN DEFAULT FALSE,
    ambient_light_sensor BOOLEAN DEFAULT FALSE,
    
    -- Risk assessment
    risk_score DECIMAL(5,4),
    fraud_history_count INTEGER DEFAULT 0,
    successful_verifications INTEGER DEFAULT 0,
    failed_verifications INTEGER DEFAULT 0,
    
    -- Usage statistics
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_sessions INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fraud patterns - รูปแบบการฉ้อโกงที่ตรวจพบ
CREATE TABLE antispoofing.fraud_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_name VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL, -- 'device_based', 'behavior_based', 'technical_based'
    
    -- Pattern description
    description TEXT,
    severity_level VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- Detection criteria
    detection_rules JSONB NOT NULL, -- Rules for detecting this pattern
    threshold_values JSONB, -- Threshold values for detection
    
    -- Pattern metadata
    confidence_weight DECIMAL(5,4) DEFAULT 0.5, -- Weight in overall fraud calculation
    false_positive_rate DECIMAL(5,4),
    detection_accuracy DECIMAL(5,4),
    
    -- Usage statistics
    detection_count INTEGER DEFAULT 0,
    last_detected_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    auto_block BOOLEAN DEFAULT FALSE, -- Automatically block when detected
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- Antispoofing checks indexes
CREATE INDEX idx_antispoofing_checks_user_id ON antispoofing.antispoofing_checks(user_id);
CREATE INDEX idx_antispoofing_checks_session_id ON antispoofing.antispoofing_checks(session_id);
CREATE INDEX idx_antispoofing_checks_check_type ON antispoofing.antispoofing_checks(check_type);
CREATE INDEX idx_antispoofing_checks_is_live ON antispoofing.antispoofing_checks(is_live);
CREATE INDEX idx_antispoofing_checks_spoof_type ON antispoofing.antispoofing_checks(spoof_type);
CREATE INDEX idx_antispoofing_checks_risk_level ON antispoofing.antispoofing_checks(risk_level);
CREATE INDEX idx_antispoofing_checks_created_at ON antispoofing.antispoofing_checks(created_at);
CREATE INDEX idx_antispoofing_checks_confidence ON antispoofing.antispoofing_checks(confidence_score);

-- Liveness sessions indexes
CREATE INDEX idx_liveness_sessions_token ON antispoofing.liveness_sessions(session_token);
CREATE INDEX idx_liveness_sessions_user_id ON antispoofing.liveness_sessions(user_id);
CREATE INDEX idx_liveness_sessions_status ON antispoofing.liveness_sessions(session_status);
CREATE INDEX idx_liveness_sessions_expires_at ON antispoofing.liveness_sessions(expires_at);
CREATE INDEX idx_liveness_sessions_created_at ON antispoofing.liveness_sessions(created_at);

-- Challenge responses indexes
CREATE INDEX idx_challenge_responses_session_id ON antispoofing.challenge_responses(session_id);
CREATE INDEX idx_challenge_responses_challenge_type ON antispoofing.challenge_responses(challenge_type);
CREATE INDEX idx_challenge_responses_success ON antispoofing.challenge_responses(challenge_success);
CREATE INDEX idx_challenge_responses_created_at ON antispoofing.challenge_responses(created_at);

-- Spoof detection models indexes
CREATE INDEX idx_spoof_models_name ON antispoofing.spoof_detection_models(model_name);
CREATE INDEX idx_spoof_models_type ON antispoofing.spoof_detection_models(model_type);
CREATE INDEX idx_spoof_models_status ON antispoofing.spoof_detection_models(status);
CREATE INDEX idx_spoof_models_is_default ON antispoofing.spoof_detection_models(is_default);

-- Device profiles indexes
CREATE INDEX idx_device_profiles_fingerprint ON antispoofing.device_profiles(device_fingerprint);
CREATE INDEX idx_device_profiles_device_type ON antispoofing.device_profiles(device_type);
CREATE INDEX idx_device_profiles_risk_score ON antispoofing.device_profiles(risk_score);
CREATE INDEX idx_device_profiles_last_seen ON antispoofing.device_profiles(last_seen_at);

-- Fraud patterns indexes
CREATE INDEX idx_fraud_patterns_pattern_type ON antispoofing.fraud_patterns(pattern_type);
CREATE INDEX idx_fraud_patterns_severity ON antispoofing.fraud_patterns(severity_level);
CREATE INDEX idx_fraud_patterns_is_active ON antispoofing.fraud_patterns(is_active);

-- Create triggers for updated_at columns
CREATE TRIGGER update_spoof_detection_models_updated_at BEFORE UPDATE ON antispoofing.spoof_detection_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_device_profiles_updated_at BEFORE UPDATE ON antispoofing.device_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fraud_patterns_updated_at BEFORE UPDATE ON antispoofing.fraud_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA antispoofing TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA antispoofing TO facesocial_user;

-- Create views for common queries

-- Recent antispoofing activity
CREATE VIEW antispoofing.recent_activity AS
SELECT 
    ac.id,
    ac.user_id,
    ac.check_type,
    ac.is_live,
    ac.confidence_score,
    ac.spoof_type,
    ac.risk_level,
    ac.request_source,
    ac.processing_time_ms,
    ac.created_at,
    u.user_id as user_reference
FROM antispoofing.antispoofing_checks ac
JOIN core.users u ON ac.user_id = u.user_id
WHERE ac.created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY ac.created_at DESC;

-- Antispoofing success rates by type
CREATE VIEW antispoofing.success_rates_by_type AS
SELECT 
    ac.check_type,
    COUNT(*) as total_checks,
    COUNT(CASE WHEN ac.is_live = true THEN 1 END) as live_detections,
    COUNT(CASE WHEN ac.is_live = false THEN 1 END) as spoof_detections,
    AVG(ac.confidence_score) as avg_confidence,
    AVG(ac.processing_time_ms) as avg_processing_time
FROM antispoofing.antispoofing_checks ac
WHERE ac.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ac.check_type
ORDER BY total_checks DESC;

-- Device risk summary
CREATE VIEW antispoofing.device_risk_summary AS
SELECT 
    dp.device_type,
    COUNT(*) as device_count,
    AVG(dp.risk_score) as avg_risk_score,
    COUNT(CASE WHEN dp.risk_score > 0.7 THEN 1 END) as high_risk_devices,
    AVG(dp.successful_verifications::DECIMAL / NULLIF(dp.total_sessions, 0)) as success_rate
FROM antispoofing.device_profiles dp
WHERE dp.last_seen_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY dp.device_type
ORDER BY avg_risk_score DESC;

-- Active liveness sessions summary
CREATE VIEW antispoofing.active_liveness_summary AS
SELECT 
    ls.session_status,
    COUNT(*) as session_count,
    AVG(ls.attempts_count) as avg_attempts,
    AVG(EXTRACT(EPOCH FROM (ls.completed_at - ls.started_at))) as avg_duration_seconds,
    COUNT(CASE WHEN ls.overall_result = true THEN 1 END) as successful_sessions
FROM antispoofing.liveness_sessions ls
WHERE ls.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY ls.session_status
ORDER BY session_count DESC;

GRANT SELECT ON antispoofing.recent_activity TO facesocial_user;
GRANT SELECT ON antispoofing.success_rates_by_type TO facesocial_user;
GRANT SELECT ON antispoofing.device_risk_summary TO facesocial_user;
GRANT SELECT ON antispoofing.active_liveness_summary TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Antispoofing tables created successfully!';
    RAISE NOTICE 'Tables: antispoofing_checks, liveness_sessions, challenge_responses, spoof_detection_models, device_profiles, fraud_patterns';
    RAISE NOTICE 'Views: recent_activity, success_rates_by_type, device_risk_summary, active_liveness_summary';
    RAISE NOTICE 'Indexes and triggers configured for optimal security and performance';
END $$;