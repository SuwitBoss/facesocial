-- Audit and Analytics Tables
-- Handles logging, monitoring, performance tracking, and analytics

-- AI operations log - บันทึกการใช้งาน AI
CREATE TABLE audit.ai_operations_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id VARCHAR(64) NOT NULL,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Operation details
    service_name VARCHAR(50) NOT NULL, -- 'face_recognition', 'deepfake_detection', etc.
    operation_type VARCHAR(50) NOT NULL, -- 'register', 'verify', 'detect', 'analyze'
    endpoint VARCHAR(100),
    http_method VARCHAR(10), -- GET, POST, PUT, DELETE
    
    -- Request details
    request_size_bytes INTEGER,
    request_headers JSONB,
    request_parameters JSONB, -- Query parameters and form data
    request_body_hash VARCHAR(64), -- Hash of request body (for privacy)
    
    -- Response details
    response_size_bytes INTEGER,
    response_status_code INTEGER,
    response_headers JSONB,
    response_body_hash VARCHAR(64), -- Hash of response body
    
    -- Processing metrics
    processing_time_ms INTEGER,
    queue_time_ms INTEGER, -- Time spent in queue
    model_inference_time_ms INTEGER, -- Time spent in AI model inference
    database_time_ms INTEGER, -- Time spent in database operations
    
    -- Resource usage
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    gpu_usage_percent DECIMAL(5,2),
    gpu_memory_usage_mb INTEGER,
    
    -- Results summary
    operation_result VARCHAR(20), -- 'success', 'error', 'partial', 'timeout'
    confidence_score DECIMAL(5,4), -- Average confidence if applicable
    items_processed INTEGER DEFAULT 1, -- Number of items processed
    
    -- Error information
    error_code VARCHAR(50),
    error_message TEXT,
    error_stack_trace TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Context and metadata
    ip_address INET,
    user_agent TEXT,
    api_key_id VARCHAR(50),
    session_id VARCHAR(128),
    request_source VARCHAR(50), -- 'web', 'mobile', 'api', 'batch'
    
    -- Geographic information
    country_code VARCHAR(2),
    region VARCHAR(50),
    city VARCHAR(100),
    
    -- Device information
    device_type VARCHAR(30), -- 'mobile', 'desktop', 'tablet', 'server'
    device_os VARCHAR(50),
    device_browser VARCHAR(50),
    
    -- Business metrics
    cost_units INTEGER DEFAULT 1, -- For billing calculations
    revenue_impact DECIMAL(10,4), -- Revenue impact of this operation
    
    -- Data quality and validation
    input_quality_score DECIMAL(5,4), -- Quality of input data
    output_quality_score DECIMAL(5,4), -- Quality of output data
    validation_passed BOOLEAN DEFAULT TRUE,
    
    -- Privacy and compliance
    contains_pii BOOLEAN DEFAULT FALSE, -- Contains personally identifiable information
    gdpr_subject BOOLEAN DEFAULT FALSE, -- Subject to GDPR
    data_retention_days INTEGER DEFAULT 365,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics - เมทริกส์ performance
CREATE TABLE audit.performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_date DATE NOT NULL,
    metric_hour INTEGER, -- 0-23, null for daily aggregates
    service_name VARCHAR(50) NOT NULL,
    
    -- Volume metrics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    timeout_requests INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_response_time_ms DECIMAL(10,2),
    p50_response_time_ms DECIMAL(10,2), -- Median
    p95_response_time_ms DECIMAL(10,2),
    p99_response_time_ms DECIMAL(10,2),
    max_response_time_ms INTEGER,
    min_response_time_ms INTEGER,
    
    -- Throughput metrics
    requests_per_second DECIMAL(10,4),
    requests_per_minute DECIMAL(10,2),
    peak_concurrent_requests INTEGER,
    
    -- AI model performance
    avg_model_inference_time_ms DECIMAL(10,2),
    avg_confidence_score DECIMAL(5,4),
    high_confidence_count INTEGER DEFAULT 0, -- > 0.9
    medium_confidence_count INTEGER DEFAULT 0, -- 0.7-0.9
    low_confidence_count INTEGER DEFAULT 0, -- < 0.7
    
    -- Resource utilization
    avg_cpu_usage DECIMAL(5,2),
    peak_cpu_usage DECIMAL(5,2),
    avg_memory_usage_mb INTEGER,
    peak_memory_usage_mb INTEGER,
    avg_gpu_usage DECIMAL(5,2),
    peak_gpu_usage DECIMAL(5,2),
    avg_gpu_memory_mb INTEGER,
    peak_gpu_memory_mb INTEGER,
    
    -- Error analysis
    error_rate DECIMAL(5,4),
    timeout_rate DECIMAL(5,4),
    model_error_count INTEGER DEFAULT 0,
    validation_error_count INTEGER DEFAULT 0,
    system_error_count INTEGER DEFAULT 0,
    
    -- Cost and efficiency
    total_cost_units INTEGER DEFAULT 0,
    cost_per_request DECIMAL(10,6),
    efficiency_score DECIMAL(5,4), -- Cost vs performance efficiency
    
    -- Data quality metrics
    avg_input_quality DECIMAL(5,4),
    avg_output_quality DECIMAL(5,4),
    data_validation_pass_rate DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(metric_date, metric_hour, service_name)
);

-- System health checks - ตรวจสอบสถานะระบบ
CREATE TABLE audit.system_health_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_name VARCHAR(100) NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    
    -- Health status
    status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy', 'critical'
    overall_score DECIMAL(5,4), -- 0.0-1.0 health score
    
    -- Response metrics
    response_time_ms INTEGER,
    response_size_bytes INTEGER,
    
    -- Specific health indicators
    cpu_usage DECIMAL(5,2),
    memory_usage_mb INTEGER,
    memory_usage_percent DECIMAL(5,2),
    disk_usage_percent DECIMAL(5,2),
    network_latency_ms INTEGER,
    
    -- GPU metrics (if applicable)
    gpu_usage DECIMAL(5,2),
    gpu_memory_usage_mb INTEGER,
    gpu_temperature DECIMAL(5,2),
    
    -- Service-specific metrics
    active_connections INTEGER,
    queue_size INTEGER,
    cache_hit_rate DECIMAL(5,4),
    database_connections INTEGER,
    
    -- Custom health indicators
    custom_metrics JSONB DEFAULT '{}',
    
    -- Check details
    check_duration_ms INTEGER,
    check_method VARCHAR(30), -- 'http', 'tcp', 'internal', 'synthetic'
    check_endpoint VARCHAR(200),
    
    -- Error information
    error_message TEXT,
    warning_messages JSONB, -- Array of warning messages
    
    -- Alerting
    alert_level VARCHAR(20), -- 'none', 'info', 'warning', 'error', 'critical'
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_recipients JSONB,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User activity logs - บันทึกการใช้งานของผู้ใช้
CREATE TABLE audit.user_activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    session_id VARCHAR(128),
    
    -- Activity details
    activity_type VARCHAR(50) NOT NULL, -- 'face_registration', 'face_verification', 'content_upload'
    activity_description TEXT,
    activity_category VARCHAR(30), -- 'authentication', 'content', 'profile', 'settings'
    
    -- Activity metadata
    resource_affected VARCHAR(100), -- What was modified/accessed
    resource_id VARCHAR(64), -- ID of the resource
    action_performed VARCHAR(50), -- 'create', 'read', 'update', 'delete', 'verify'
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    referer_url TEXT,
    
    -- Geographic information
    country_code VARCHAR(2),
    region VARCHAR(50),
    city VARCHAR(100),
    timezone VARCHAR(50),
    
    -- Device information
    device_fingerprint VARCHAR(128),
    device_type VARCHAR(30),
    device_os VARCHAR(50),
    device_browser VARCHAR(50),
    screen_resolution VARCHAR(20),
    
    -- Security context
    authentication_method VARCHAR(30), -- 'password', 'face_recognition', 'oauth', 'api_key'
    risk_score DECIMAL(5,4), -- Calculated risk score for this activity
    anomaly_indicators JSONB, -- Detected anomalies
    
    -- Result and impact
    activity_result VARCHAR(20), -- 'success', 'failure', 'partial', 'blocked'
    impact_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    
    -- Privacy and compliance
    data_accessed BOOLEAN DEFAULT FALSE,
    data_modified BOOLEAN DEFAULT FALSE,
    pii_involved BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security events - เหตุการณ์ด้านความปลอดภัย
CREATE TABLE audit.security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(64) NOT NULL UNIQUE,
    
    -- Event classification
    event_type VARCHAR(50) NOT NULL, -- 'failed_authentication', 'suspicious_activity', 'data_breach_attempt'
    severity_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    event_category VARCHAR(30) NOT NULL, -- 'authentication', 'authorization', 'data_access', 'system'
    
    -- Affected entities
    user_id UUID REFERENCES core.users(user_id),
    affected_resource VARCHAR(100),
    resource_id VARCHAR(64),
    
    -- Event details
    event_description TEXT NOT NULL,
    event_source VARCHAR(50), -- Which system/service detected this
    detection_method VARCHAR(50), -- How it was detected
    
    -- Attack/threat information
    attack_type VARCHAR(50), -- 'brute_force', 'sql_injection', 'spoofing', 'deepfake'
    attack_vector VARCHAR(50), -- 'web', 'api', 'mobile', 'internal'
    threat_indicators JSONB, -- IOCs and threat indicators
    
    -- Source information
    source_ip INET,
    source_country VARCHAR(50),
    source_user_agent TEXT,
    source_fingerprint VARCHAR(128),
    
    -- Impact assessment
    impact_description TEXT,
    data_compromised BOOLEAN DEFAULT FALSE,
    systems_affected JSONB, -- List of affected systems
    estimated_impact_score DECIMAL(5,4),
    
    -- Response actions
    response_actions JSONB, -- Actions taken in response
    blocked_ip BOOLEAN DEFAULT FALSE,
    user_suspended BOOLEAN DEFAULT FALSE,
    investigation_required BOOLEAN DEFAULT FALSE,
    
    -- Investigation details
    assigned_to VARCHAR(100), -- Who is investigating
    investigation_status VARCHAR(30), -- 'open', 'in_progress', 'resolved', 'closed'
    investigation_notes TEXT,
    
    -- Resolution
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_description TEXT,
    false_positive BOOLEAN DEFAULT FALSE,
    
    -- Compliance and reporting
    regulatory_reporting_required BOOLEAN DEFAULT FALSE,
    gdpr_breach BOOLEAN DEFAULT FALSE,
    breach_notification_sent BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data retention policies - นโยบายการเก็บข้อมูล
CREATE TABLE audit.data_retention_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Scope
    applies_to_table VARCHAR(100) NOT NULL,
    applies_to_schema VARCHAR(50),
    data_category VARCHAR(50), -- 'logs', 'user_data', 'analytics', 'security'
    
    -- Retention rules
    retention_period_days INTEGER NOT NULL,
    archive_after_days INTEGER, -- Move to archive after X days
    delete_after_days INTEGER, -- Permanently delete after X days
    
    -- Conditions
    retention_conditions JSONB, -- Conditions for applying this policy
    exceptions JSONB, -- Exceptions to the policy
    
    -- Actions
    automated_cleanup BOOLEAN DEFAULT TRUE,
    cleanup_method VARCHAR(30), -- 'delete', 'archive', 'anonymize'
    notification_before_cleanup INTEGER DEFAULT 7, -- Days before cleanup to notify
    
    -- Compliance
    legal_hold_override BOOLEAN DEFAULT FALSE, -- Can legal hold override this policy
    gdpr_compliant BOOLEAN DEFAULT TRUE,
    regulatory_requirements JSONB,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_executed_at TIMESTAMP WITH TIME ZONE,
    next_execution_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- AI operations log indexes
CREATE INDEX idx_ai_operations_log_user_id ON audit.ai_operations_log(user_id);
CREATE INDEX idx_ai_operations_log_service ON audit.ai_operations_log(service_name);
CREATE INDEX idx_ai_operations_log_operation ON audit.ai_operations_log(operation_type);
CREATE INDEX idx_ai_operations_log_created_at ON audit.ai_operations_log(created_at);
CREATE INDEX idx_ai_operations_log_result ON audit.ai_operations_log(operation_result);
CREATE INDEX idx_ai_operations_log_processing_time ON audit.ai_operations_log(processing_time_ms);
CREATE INDEX idx_ai_operations_log_ip_address ON audit.ai_operations_log(ip_address);

-- Performance metrics indexes
CREATE INDEX idx_performance_metrics_date ON audit.performance_metrics(metric_date);
CREATE INDEX idx_performance_metrics_service ON audit.performance_metrics(service_name);
CREATE INDEX idx_performance_metrics_hour ON audit.performance_metrics(metric_hour);
CREATE INDEX idx_performance_metrics_error_rate ON audit.performance_metrics(error_rate);

-- System health checks indexes
CREATE INDEX idx_system_health_checks_service ON audit.system_health_checks(service_name);
CREATE INDEX idx_system_health_checks_status ON audit.system_health_checks(status);
CREATE INDEX idx_system_health_checks_timestamp ON audit.system_health_checks(timestamp);
CREATE INDEX idx_system_health_checks_alert_level ON audit.system_health_checks(alert_level);

-- User activity logs indexes
CREATE INDEX idx_user_activity_logs_user_id ON audit.user_activity_logs(user_id);
CREATE INDEX idx_user_activity_logs_activity_type ON audit.user_activity_logs(activity_type);
CREATE INDEX idx_user_activity_logs_created_at ON audit.user_activity_logs(created_at);
CREATE INDEX idx_user_activity_logs_ip_address ON audit.user_activity_logs(ip_address);
CREATE INDEX idx_user_activity_logs_risk_score ON audit.user_activity_logs(risk_score);

-- Security events indexes
CREATE INDEX idx_security_events_event_type ON audit.security_events(event_type);
CREATE INDEX idx_security_events_severity ON audit.security_events(severity_level);
CREATE INDEX idx_security_events_user_id ON audit.security_events(user_id);
CREATE INDEX idx_security_events_detected_at ON audit.security_events(detected_at);
CREATE INDEX idx_security_events_source_ip ON audit.security_events(source_ip);
CREATE INDEX idx_security_events_investigation_status ON audit.security_events(investigation_status);

-- Data retention policies indexes
CREATE INDEX idx_data_retention_policies_table ON audit.data_retention_policies(applies_to_table);
CREATE INDEX idx_data_retention_policies_is_active ON audit.data_retention_policies(is_active);
CREATE INDEX idx_data_retention_policies_next_execution ON audit.data_retention_policies(next_execution_at);

-- Create triggers for updated_at columns
CREATE TRIGGER update_security_events_updated_at BEFORE UPDATE ON audit.security_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_retention_policies_updated_at BEFORE UPDATE ON audit.data_retention_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA audit TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA audit TO facesocial_user;

-- Create partitioned tables for high-volume logs (optional optimization)
-- This creates monthly partitions for the ai_operations_log table

CREATE TABLE audit.ai_operations_log_y2024m01 PARTITION OF audit.ai_operations_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE audit.ai_operations_log_y2024m02 PARTITION OF audit.ai_operations_log
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Function to create monthly partitions automatically
CREATE OR REPLACE FUNCTION audit.create_monthly_partition(table_name TEXT, schema_name TEXT DEFAULT 'audit')
RETURNS BOOLEAN AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
    sql_statement TEXT;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE);
    end_date := start_date + INTERVAL '1 month';
    partition_name := format('%s_y%sm%s', 
                           table_name, 
                           to_char(start_date, 'YYYY'), 
                           to_char(start_date, 'MM'));
    
    sql_statement := format('CREATE TABLE IF NOT EXISTS %I.%I PARTITION OF %I.%I FOR VALUES FROM (%L) TO (%L)',
                          schema_name, partition_name, schema_name, table_name, start_date, end_date);
    
    EXECUTE sql_statement;
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Create views for common audit queries

-- Daily AI operations summary
CREATE VIEW audit.daily_operations_summary AS
SELECT 
    DATE(aol.created_at) as operation_date,
    aol.service_name,
    COUNT(*) as total_operations,
    COUNT(CASE WHEN aol.operation_result = 'success' THEN 1 END) as successful_operations,
    COUNT(CASE WHEN aol.operation_result = 'error' THEN 1 END) as failed_operations,
    AVG(aol.processing_time_ms) as avg_processing_time,
    AVG(aol.confidence_score) as avg_confidence_score,
    SUM(aol.cost_units) as total_cost_units
FROM audit.ai_operations_log aol
WHERE aol.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(aol.created_at), aol.service_name
ORDER BY operation_date DESC, aol.service_name;

-- System health summary
CREATE VIEW audit.system_health_summary AS
SELECT 
    shc.service_name,
    shc.status,
    COUNT(*) as check_count,
    AVG(shc.overall_score) as avg_health_score,
    AVG(shc.response_time_ms) as avg_response_time,
    MAX(shc.timestamp) as last_check_time
FROM audit.system_health_checks shc
WHERE shc.timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY shc.service_name, shc.status
ORDER BY shc.service_name, shc.status;

-- Security incidents summary
CREATE VIEW audit.security_incidents_summary AS
SELECT 
    DATE(se.detected_at) as incident_date,
    se.event_type,
    se.severity_level,
    COUNT(*) as incident_count,
    COUNT(CASE WHEN se.investigation_status = 'resolved' THEN 1 END) as resolved_count,
    COUNT(CASE WHEN se.false_positive = true THEN 1 END) as false_positive_count,
    AVG(EXTRACT(EPOCH FROM (se.resolved_at - se.detected_at))/3600) as avg_resolution_hours
FROM audit.security_events se
WHERE se.detected_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(se.detected_at), se.event_type, se.severity_level
ORDER BY incident_date DESC, se.severity_level, se.event_type;

-- User activity patterns
CREATE VIEW audit.user_activity_patterns AS
SELECT 
    EXTRACT(HOUR FROM ual.created_at) as hour_of_day,
    ual.activity_type,
    COUNT(*) as activity_count,
    COUNT(DISTINCT ual.user_id) as unique_users,
    AVG(ual.risk_score) as avg_risk_score,
    COUNT(CASE WHEN ual.activity_result = 'success' THEN 1 END) as successful_activities
FROM audit.user_activity_logs ual
WHERE ual.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY EXTRACT(HOUR FROM ual.created_at), ual.activity_type
ORDER BY hour_of_day, ual.activity_type;

GRANT SELECT ON audit.daily_operations_summary TO facesocial_user;
GRANT SELECT ON audit.system_health_summary TO facesocial_user;
GRANT SELECT ON audit.security_incidents_summary TO facesocial_user;
GRANT SELECT ON audit.user_activity_patterns TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Audit and Analytics tables created successfully!';
    RAISE NOTICE 'Tables: ai_operations_log, performance_metrics, system_health_checks, user_activity_logs, security_events, data_retention_policies';
    RAISE NOTICE 'Views: daily_operations_summary, system_health_summary, security_incidents_summary, user_activity_patterns';
    RAISE NOTICE 'Partitioning and automated functions configured for high-volume logging';
    RAISE NOTICE 'Security monitoring and compliance tracking enabled';
END $$;