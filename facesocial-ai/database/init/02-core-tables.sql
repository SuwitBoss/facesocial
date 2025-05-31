-- Core Tables for FaceSocial AI System
-- These tables form the foundation for all AI services

-- Users table (references main user service)
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE, -- Reference to main user service
    face_id UUID, -- Primary face for this user
    
    -- Privacy and consent settings
    privacy_settings JSONB DEFAULT '{}',
    consent_given BOOLEAN DEFAULT FALSE,
    consent_timestamp TIMESTAMP WITH TIME ZONE,
    consent_version VARCHAR(10) DEFAULT '1.0',
    
    -- AI service preferences
    ai_settings JSONB DEFAULT '{
        "face_recognition_enabled": true,
        "antispoofing_enabled": true,
        "deepfake_detection_enabled": true,
        "demographics_analysis_enabled": false,
        "data_retention_days": 365
    }',
    
    -- Account status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'suspended', 'deleted'
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- API Keys for service authentication
CREATE TABLE core.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id VARCHAR(32) NOT NULL UNIQUE DEFAULT generate_short_uuid(),
    key_hash VARCHAR(256) NOT NULL, -- bcrypt hash of the actual key
    
    -- Key metadata
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Permissions and limits
    permissions JSONB DEFAULT '[]', -- Array of allowed endpoints/services
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    
    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    total_requests INTEGER DEFAULT 0,
    
    -- Key lifecycle
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Ownership
    created_by UUID REFERENCES core.users(user_id),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Service sessions (for tracking user sessions across AI services)
CREATE TABLE core.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(128) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Session metadata
    session_type VARCHAR(50) NOT NULL, -- 'web', 'mobile', 'api', 'batch'
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Session data
    session_data JSONB DEFAULT '{}',
    
    -- AI services used in this session
    services_used JSONB DEFAULT '[]',
    
    -- Session status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'expired', 'terminated'
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours')
);

-- Image metadata (for tracking processed images across services)
CREATE TABLE core.image_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash
    
    -- Image properties
    width INTEGER,
    height INTEGER,
    format VARCHAR(10), -- 'JPEG', 'PNG', 'WEBP', etc.
    file_size_bytes INTEGER,
    color_space VARCHAR(20), -- 'RGB', 'GRAYSCALE', etc.
    
    -- Image quality metrics
    quality_score DECIMAL(5,4), -- 0.0000-1.0000
    sharpness_score DECIMAL(5,4),
    brightness_score DECIMAL(5,4),
    contrast_score DECIMAL(5,4),
    
    -- Storage information
    storage_url TEXT,
    storage_provider VARCHAR(50), -- 'local', 'aws_s3', 'gcp_storage', etc.
    
    -- Processing history
    services_processed JSONB DEFAULT '[]', -- Array of services that processed this image
    processing_count INTEGER DEFAULT 0,
    
    -- Data retention
    retention_policy VARCHAR(50) DEFAULT 'standard', -- 'temporary', 'standard', 'long_term'
    auto_delete_at TIMESTAMP WITH TIME ZONE,
    
    -- Ownership and privacy
    uploaded_by UUID REFERENCES core.users(user_id),
    is_public BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Request logs (for all AI service requests)
CREATE TABLE core.request_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(64) NOT NULL UNIQUE,
    
    -- Request details
    service_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    
    -- User context
    user_id UUID REFERENCES core.users(user_id),
    session_id VARCHAR(128) REFERENCES core.sessions(session_id),
    api_key_id VARCHAR(32) REFERENCES core.api_keys(key_id),
    
    -- Request metadata
    ip_address INET,
    user_agent TEXT,
    request_headers JSONB,
    request_size_bytes INTEGER,
    
    -- Response details
    status_code INTEGER,
    response_size_bytes INTEGER,
    processing_time_ms INTEGER,
    
    -- Error information
    error_code VARCHAR(50),
    error_message TEXT,
    
    -- Billing and usage
    billing_units INTEGER DEFAULT 1,
    cost_usd DECIMAL(10,6),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System health metrics
CREATE TABLE core.system_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    
    -- Health metrics
    status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    response_time_ms INTEGER,
    error_rate DECIMAL(5,4),
    cpu_usage DECIMAL(5,2),
    memory_usage_mb INTEGER,
    gpu_usage DECIMAL(5,2),
    gpu_memory_mb INTEGER,
    
    -- Service-specific metrics
    active_connections INTEGER,
    queue_size INTEGER,
    processed_requests_1min INTEGER,
    
    -- Additional metrics (JSON format for flexibility)
    custom_metrics JSONB DEFAULT '{}',
    
    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance

-- Users table indexes
CREATE INDEX idx_users_user_id ON core.users(user_id);
CREATE INDEX idx_users_face_id ON core.users(face_id);
CREATE INDEX idx_users_status ON core.users(status);
CREATE INDEX idx_users_created_at ON core.users(created_at);

-- API Keys indexes
CREATE INDEX idx_api_keys_key_id ON core.api_keys(key_id);
CREATE INDEX idx_api_keys_is_active ON core.api_keys(is_active);
CREATE INDEX idx_api_keys_expires_at ON core.api_keys(expires_at);
CREATE INDEX idx_api_keys_created_by ON core.api_keys(created_by);

-- Sessions indexes
CREATE INDEX idx_sessions_session_id ON core.sessions(session_id);
CREATE INDEX idx_sessions_user_id ON core.sessions(user_id);
CREATE INDEX idx_sessions_status ON core.sessions(status);
CREATE INDEX idx_sessions_expires_at ON core.sessions(expires_at);
CREATE INDEX idx_sessions_created_at ON core.sessions(created_at);

-- Image metadata indexes
CREATE INDEX idx_image_metadata_hash ON core.image_metadata(image_hash);
CREATE INDEX idx_image_metadata_uploaded_by ON core.image_metadata(uploaded_by);
CREATE INDEX idx_image_metadata_format ON core.image_metadata(format);
CREATE INDEX idx_image_metadata_created_at ON core.image_metadata(created_at);
CREATE INDEX idx_image_metadata_auto_delete_at ON core.image_metadata(auto_delete_at);

-- Request logs indexes
CREATE INDEX idx_request_logs_request_id ON core.request_logs(request_id);
CREATE INDEX idx_request_logs_service_name ON core.request_logs(service_name);
CREATE INDEX idx_request_logs_user_id ON core.request_logs(user_id);
CREATE INDEX idx_request_logs_status_code ON core.request_logs(status_code);
CREATE INDEX idx_request_logs_created_at ON core.request_logs(created_at);

-- System health indexes
CREATE INDEX idx_system_health_service_name ON core.system_health(service_name);
CREATE INDEX idx_system_health_status ON core.system_health(status);
CREATE INDEX idx_system_health_recorded_at ON core.system_health(recorded_at);

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON core.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON core.api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_image_metadata_updated_at BEFORE UPDATE ON core.image_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to application user
GRANT ALL ON ALL TABLES IN SCHEMA core TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA core TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Core tables created successfully!';
    RAISE NOTICE 'Tables: users, api_keys, sessions, image_metadata, request_logs, system_health';
    RAISE NOTICE 'Indexes and triggers configured';
END $$;