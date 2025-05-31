-- Configuration and Settings Tables
-- Handles system configuration, model management, and service settings

-- AI models - การจัดการโมเดล AI
CREATE TABLE config.ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Model details
    model_type VARCHAR(50) NOT NULL, -- 'face_detection', 'face_recognition', 'antispoofing', etc.
    model_path TEXT NOT NULL,
    model_url TEXT, -- URL to download model if not local
    model_size_mb DECIMAL(10,2),
    model_format VARCHAR(20) DEFAULT 'onnx', -- 'onnx', 'pytorch', 'tensorflow', 'tflite'
    
    -- Model architecture
    architecture VARCHAR(50), -- 'yolo', 'resnet', 'efficientnet', 'transformer'
    input_dimensions JSONB, -- {width, height, channels}
    output_dimensions JSONB, -- Output format description
    preprocessing_config JSONB, -- Preprocessing configuration
    postprocessing_config JSONB, -- Postprocessing configuration
    
    -- Performance characteristics
    avg_inference_time_ms DECIMAL(10,2),
    accuracy_score DECIMAL(5,4),
    benchmark_scores JSONB, -- Various benchmark results
    supported_formats JSONB, -- Supported input formats
    
    -- Hardware requirements
    min_gpu_memory_mb INTEGER,
    min_cpu_cores INTEGER,
    supports_batch_processing BOOLEAN DEFAULT TRUE,
    max_batch_size INTEGER DEFAULT 1,
    optimal_batch_size INTEGER DEFAULT 1,
    
    -- Model capabilities
    capabilities JSONB, -- What the model can do
    limitations JSONB, -- Known limitations
    use_cases JSONB, -- Recommended use cases
    
    -- Deployment configuration
    deployment_config JSONB, -- Deployment-specific settings
    environment_requirements JSONB, -- Environment dependencies
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing', 'maintenance'
    is_default BOOLEAN DEFAULT FALSE,
    priority INTEGER DEFAULT 0, -- Higher priority models are preferred
    
    -- Versioning and updates
    parent_model_id UUID REFERENCES config.ai_models(id), -- Previous version
    changelog TEXT, -- What changed in this version
    migration_notes TEXT, -- Notes for migrating from previous version
    
    -- Metadata
    description TEXT,
    training_date DATE,
    deployment_date DATE,
    deprecation_date DATE,
    
    -- Legal and compliance
    license_type VARCHAR(50), -- 'open_source', 'commercial', 'proprietary'
    license_text TEXT,
    training_data_source TEXT,
    bias_assessment JSONB, -- Bias assessment results
    ethical_considerations TEXT,
    
    -- Monitoring and alerts
    monitoring_enabled BOOLEAN DEFAULT TRUE,
    alert_thresholds JSONB, -- Thresholds for alerts
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(service_name, model_name, model_version)
);

-- Service settings - การตั้งค่าแต่ละ service
CREATE TABLE config.service_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    setting_category VARCHAR(50) NOT NULL, -- 'performance', 'security', 'features', 'limits'
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB NOT NULL,
    setting_type VARCHAR(20) NOT NULL, -- 'string', 'number', 'boolean', 'object', 'array'
    
    -- Setting metadata
    display_name VARCHAR(100),
    description TEXT,
    default_value JSONB,
    allowed_values JSONB, -- Allowed values for validation
    validation_rules JSONB, -- Validation rules
    
    -- Access control
    is_user_configurable BOOLEAN DEFAULT FALSE,
    is_admin_only BOOLEAN DEFAULT TRUE,
    requires_restart BOOLEAN DEFAULT FALSE,
    requires_authentication BOOLEAN DEFAULT TRUE,
    
    -- Environment specific
    environment VARCHAR(20) DEFAULT 'all', -- 'development', 'staging', 'production', 'all'
    applies_to_regions JSONB, -- Geographic regions this applies to
    
    -- Change management
    effective_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    effective_until TIMESTAMP WITH TIME ZONE,
    previous_value JSONB, -- Previous value for rollback
    change_reason TEXT,
    changed_by VARCHAR(100),
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_sensitive BOOLEAN DEFAULT FALSE, -- Contains sensitive information
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(service_name, setting_category, setting_key, environment)
);

-- Feature flags - การควบคุม features
CREATE TABLE config.feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_name VARCHAR(100) NOT NULL UNIQUE,
    flag_key VARCHAR(100) NOT NULL UNIQUE, -- Key used in code
    
    -- Flag details
    description TEXT,
    flag_type VARCHAR(30) DEFAULT 'boolean', -- 'boolean', 'string', 'number', 'percentage'
    default_value JSONB NOT NULL,
    
    -- Status
    is_enabled BOOLEAN DEFAULT FALSE,
    is_permanent BOOLEAN DEFAULT FALSE, -- Cannot be changed without deployment
    
    -- Targeting
    target_users JSONB, -- Specific users to enable for
    target_groups JSONB, -- User groups to enable for
    target_percentage DECIMAL(5,2) DEFAULT 0, -- Percentage of users to enable for
    targeting_rules JSONB, -- Complex targeting rules
    
    -- Service scope
    services JSONB, -- Which services this flag applies to
    environments JSONB DEFAULT '["all"]', -- Which environments
    regions JSONB, -- Geographic regions
    
    -- Lifecycle
    created_by VARCHAR(100),
    enabled_at TIMESTAMP WITH TIME ZONE,
    disabled_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Monitoring
    usage_tracking BOOLEAN DEFAULT TRUE,
    metrics_collection BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API rate limits - การจำกัดอัตรา API
CREATE TABLE config.api_rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    limit_name VARCHAR(100) NOT NULL,
    
    -- Scope
    applies_to VARCHAR(30) NOT NULL, -- 'global', 'user', 'api_key', 'ip', 'endpoint'
    scope_value VARCHAR(200), -- Specific value (user_id, api_key, etc.)
    endpoint_pattern VARCHAR(200), -- Endpoint pattern this applies to
    
    -- Limit configuration
    requests_per_minute INTEGER,
    requests_per_hour INTEGER,
    requests_per_day INTEGER,
    requests_per_month INTEGER,
    concurrent_requests INTEGER, -- Max concurrent requests
    
    -- Burst allowance
    burst_capacity INTEGER, -- Allow bursts up to this limit
    burst_refill_rate INTEGER, -- How fast burst capacity refills
    
    -- Actions when limit exceeded
    action_on_exceed VARCHAR(30) DEFAULT 'throttle', -- 'block', 'throttle', 'warn'
    block_duration_minutes INTEGER DEFAULT 60,
    warning_threshold_percentage INTEGER DEFAULT 80,
    
    -- Priority and overrides
    priority INTEGER DEFAULT 0, -- Higher priority limits override lower ones
    can_override BOOLEAN DEFAULT FALSE, -- Can be overridden by higher priority limits
    
    -- Status and scheduling
    is_active BOOLEAN DEFAULT TRUE,
    effective_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    effective_until TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    description TEXT,
    reason TEXT,
    created_by VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System parameters - พารามิเตอร์ระบบ
CREATE TABLE config.system_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameter_name VARCHAR(100) NOT NULL UNIQUE,
    parameter_category VARCHAR(50) NOT NULL, -- 'performance', 'security', 'business', 'technical'
    
    -- Parameter value
    parameter_value JSONB NOT NULL,
    parameter_type VARCHAR(20) NOT NULL, -- 'string', 'number', 'boolean', 'object'
    
    -- Validation
    min_value JSONB,
    max_value JSONB,
    allowed_values JSONB,
    validation_regex VARCHAR(500),
    
    -- Metadata
    display_name VARCHAR(100),
    description TEXT,
    unit VARCHAR(20), -- Unit of measurement
    default_value JSONB,
    
    -- Change management
    requires_restart BOOLEAN DEFAULT FALSE,
    requires_admin_approval BOOLEAN DEFAULT TRUE,
    change_impact_assessment TEXT,
    
    -- Monitoring
    monitor_changes BOOLEAN DEFAULT TRUE,
    alert_on_change BOOLEAN DEFAULT FALSE,
    
    -- History
    previous_value JSONB,
    last_changed_by VARCHAR(100),
    last_changed_at TIMESTAMP WITH TIME ZONE,
    change_reason TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configuration templates - เทมเพลตการตั้งค่า
CREATE TABLE config.configuration_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(100) NOT NULL,
    template_category VARCHAR(50) NOT NULL, -- 'deployment', 'service', 'security', 'performance'
    
    -- Template details
    description TEXT,
    use_case TEXT,
    target_environment VARCHAR(30), -- 'development', 'staging', 'production'
    
    -- Configuration data
    configuration JSONB NOT NULL, -- The actual configuration
    variables JSONB, -- Template variables that can be substituted
    
    -- Dependencies
    required_services JSONB, -- Services that must be available
    required_models JSONB, -- Models that must be present
    required_features JSONB, -- Features that must be enabled
    
    -- Validation
    validation_schema JSONB, -- JSON schema for validation
    test_cases JSONB, -- Test cases to validate the configuration
    
    -- Versioning
    version VARCHAR(20) DEFAULT '1.0',
    parent_template_id UUID REFERENCES config.configuration_templates(id),
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    author VARCHAR(100),
    tags JSONB, -- Tags for categorization
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(template_name, version)
);

-- Environment configurations - การตั้งค่าแต่ละ environment
CREATE TABLE config.environment_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    environment_name VARCHAR(30) NOT NULL UNIQUE, -- 'development', 'staging', 'production'
    
    -- Environment details
    description TEXT,
    is_production BOOLEAN DEFAULT FALSE,
    
    -- Configuration overrides
    config_overrides JSONB DEFAULT '{}', -- Configuration overrides for this environment
    feature_flags JSONB DEFAULT '{}', -- Feature flag overrides
    
    -- Resource limits
    max_cpu_cores INTEGER,
    max_memory_gb INTEGER,
    max_gpu_memory_gb INTEGER,
    max_storage_gb INTEGER,
    max_concurrent_users INTEGER,
    
    -- Security settings
    security_level VARCHAR(20) DEFAULT 'standard', -- 'minimal', 'standard', 'enhanced', 'maximum'
    encryption_required BOOLEAN DEFAULT TRUE,
    audit_level VARCHAR(20) DEFAULT 'standard', -- 'minimal', 'standard', 'detailed'
    
    -- Monitoring and alerting
    monitoring_enabled BOOLEAN DEFAULT TRUE,
    alerting_enabled BOOLEAN DEFAULT TRUE,
    log_level VARCHAR(20) DEFAULT 'INFO',
    metrics_retention_days INTEGER DEFAULT 30,
    
    -- Deployment settings
    auto_deployment BOOLEAN DEFAULT FALSE,
    rollback_enabled BOOLEAN DEFAULT TRUE,
    health_check_interval_seconds INTEGER DEFAULT 30,
    
    -- Data retention
    default_data_retention_days INTEGER DEFAULT 90,
    pii_retention_days INTEGER DEFAULT 30,
    log_retention_days INTEGER DEFAULT 7,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    maintenance_mode BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configuration change log - บันทึกการเปลี่ยนแปลงการตั้งค่า
CREATE TABLE config.configuration_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    change_id VARCHAR(64) NOT NULL UNIQUE,
    
    -- Change details
    change_type VARCHAR(30) NOT NULL, -- 'model_update', 'setting_change', 'feature_flag', 'parameter_change'
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    
    -- What changed
    field_name VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    change_summary TEXT,
    
    -- Change context
    change_reason TEXT,
    change_impact_assessment TEXT,
    rollback_instructions TEXT,
    
    -- Authorization
    changed_by VARCHAR(100) NOT NULL,
    approved_by VARCHAR(100),
    approval_required BOOLEAN DEFAULT FALSE,
    approval_status VARCHAR(20) DEFAULT 'approved', -- 'pending', 'approved', 'rejected'
    
    -- Deployment
    deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployment_method VARCHAR(30), -- 'automatic', 'manual', 'scheduled'
    
    -- Rollback capability
    can_rollback BOOLEAN DEFAULT TRUE,
    rollback_executed BOOLEAN DEFAULT FALSE,
    rollback_at TIMESTAMP WITH TIME ZONE,
    rollback_by VARCHAR(100),
    
    -- Change tracking
    change_request_id VARCHAR(64), -- External change request ID
    related_changes JSONB, -- Related configuration changes
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- AI models indexes
CREATE INDEX idx_ai_models_service ON config.ai_models(service_name);
CREATE INDEX idx_ai_models_model_type ON config.ai_models(model_type);
CREATE INDEX idx_ai_models_status ON config.ai_models(status);
CREATE INDEX idx_ai_models_is_default ON config.ai_models(is_default);
CREATE INDEX idx_ai_models_priority ON config.ai_models(priority DESC);

-- Service settings indexes
CREATE INDEX idx_service_settings_service ON config.service_settings(service_name);
CREATE INDEX idx_service_settings_category ON config.service_settings(setting_category);
CREATE INDEX idx_service_settings_environment ON config.service_settings(environment);
CREATE INDEX idx_service_settings_is_active ON config.service_settings(is_active);

-- Feature flags indexes
CREATE INDEX idx_feature_flags_flag_key ON config.feature_flags(flag_key);
CREATE INDEX idx_feature_flags_is_enabled ON config.feature_flags(is_enabled);
CREATE INDEX idx_feature_flags_expires_at ON config.feature_flags(expires_at);

-- API rate limits indexes
CREATE INDEX idx_api_rate_limits_applies_to ON config.api_rate_limits(applies_to);
CREATE INDEX idx_api_rate_limits_scope_value ON config.api_rate_limits(scope_value);
CREATE INDEX idx_api_rate_limits_is_active ON config.api_rate_limits(is_active);
CREATE INDEX idx_api_rate_limits_priority ON config.api_rate_limits(priority DESC);

-- System parameters indexes
CREATE INDEX idx_system_parameters_category ON config.system_parameters(parameter_category);
CREATE INDEX idx_system_parameters_name ON config.system_parameters(parameter_name);

-- Configuration templates indexes
CREATE INDEX idx_configuration_templates_category ON config.configuration_templates(template_category);
CREATE INDEX idx_configuration_templates_environment ON config.configuration_templates(target_environment);
CREATE INDEX idx_configuration_templates_is_active ON config.configuration_templates(is_active);

-- Environment configs indexes
CREATE INDEX idx_environment_configs_name ON config.environment_configs(environment_name);
CREATE INDEX idx_environment_configs_is_production ON config.environment_configs(is_production);
CREATE INDEX idx_environment_configs_is_active ON config.environment_configs(is_active);

-- Configuration changes indexes
CREATE INDEX idx_configuration_changes_change_type ON config.configuration_changes(change_type);
CREATE INDEX idx_configuration_changes_table_name ON config.configuration_changes(table_name);
CREATE INDEX idx_configuration_changes_changed_by ON config.configuration_changes(changed_by);
CREATE INDEX idx_configuration_changes_created_at ON config.configuration_changes(created_at);
CREATE INDEX idx_configuration_changes_deployed ON config.configuration_changes(deployed);

-- Create triggers for updated_at columns
CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON config.ai_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_service_settings_updated_at BEFORE UPDATE ON config.service_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_flags_updated_at BEFORE UPDATE ON config.feature_flags
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_rate_limits_updated_at BEFORE UPDATE ON config.api_rate_limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_parameters_updated_at BEFORE UPDATE ON config.system_parameters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configuration_templates_updated_at BEFORE UPDATE ON config.configuration_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_environment_configs_updated_at BEFORE UPDATE ON config.environment_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger to log configuration changes
CREATE OR REPLACE FUNCTION config.log_configuration_change()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO config.configuration_changes (
        change_id,
        change_type,
        table_name,
        record_id,
        old_value,
        new_value,
        changed_by
    ) VALUES (
        generate_short_uuid(),
        TG_TABLE_NAME,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE row_to_json(OLD) END,
        CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE row_to_json(NEW) END,
        current_user
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply configuration change logging to important tables
CREATE TRIGGER log_ai_models_changes AFTER INSERT OR UPDATE OR DELETE ON config.ai_models
    FOR EACH ROW EXECUTE FUNCTION config.log_configuration_change();

CREATE TRIGGER log_service_settings_changes AFTER INSERT OR UPDATE OR DELETE ON config.service_settings
    FOR EACH ROW EXECUTE FUNCTION config.log_configuration_change();

CREATE TRIGGER log_feature_flags_changes AFTER INSERT OR UPDATE OR DELETE ON config.feature_flags
    FOR EACH ROW EXECUTE FUNCTION config.log_configuration_change();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA config TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA config TO facesocial_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA config TO facesocial_user;

-- Insert default configurations

-- Default environments
INSERT INTO config.environment_configs (environment_name, description, is_production, security_level) VALUES
('development', 'Development environment', FALSE, 'minimal'),
('staging', 'Staging environment for testing', FALSE, 'standard'),
('production', 'Production environment', TRUE, 'maximum');

-- Default system parameters
INSERT INTO config.system_parameters (parameter_name, parameter_category, parameter_value, parameter_type, description) VALUES
('max_concurrent_requests', 'performance', '100', 'number', 'Maximum concurrent requests per service'),
('default_timeout_seconds', 'performance', '30', 'number', 'Default timeout for API requests'),
('max_upload_size_mb', 'performance', '100', 'number', 'Maximum upload size in MB'),
('enable_metrics_collection', 'monitoring', 'true', 'boolean', 'Enable metrics collection'),
('log_level', 'logging', '"INFO"', 'string', 'Default log level'),
('session_timeout_minutes', 'security', '60', 'number', 'Session timeout in minutes'),
('max_failed_login_attempts', 'security', '5', 'number', 'Maximum failed login attempts before lockout'),
('encryption_algorithm', 'security', '"AES-256"', 'string', 'Default encryption algorithm');

-- Default API rate limits
INSERT INTO config.api_rate_limits (limit_name, applies_to, requests_per_minute, requests_per_hour, requests_per_day, description) VALUES
('global_default', 'global', 1000, 10000, 100000, 'Global default rate limit'),
('user_default', 'user', 60, 1000, 10000, 'Default rate limit per user'),
('ip_default', 'ip', 100, 2000, 20000, 'Default rate limit per IP address');

-- Create views for common configuration queries

-- Active models by service
CREATE VIEW config.active_models_by_service AS
SELECT 
    am.service_name,
    am.model_type,
    am.model_name,
    am.model_version,
    am.is_default,
    am.priority,
    am.avg_inference_time_ms,
    am.accuracy_score
FROM config.ai_models am
WHERE am.status = 'active'
ORDER BY am.service_name, am.model_type, am.priority DESC, am.is_default DESC;

-- Current environment settings
CREATE VIEW config.current_environment_settings AS
SELECT 
    ec.environment_name,
    ec.is_production,
    ec.security_level,
    ec.monitoring_enabled,
    ec.is_active,
    ec.maintenance_mode
FROM config.environment_configs ec
WHERE ec.is_active = TRUE;

-- Active feature flags
CREATE VIEW config.active_feature_flags AS
SELECT 
    ff.flag_name,
    ff.flag_key,
    ff.is_enabled,
    ff.target_percentage,
    ff.services,
    ff.environments,
    ff.expires_at
FROM config.feature_flags ff
WHERE ff.is_enabled = TRUE
AND (ff.expires_at IS NULL OR ff.expires_at > CURRENT_TIMESTAMP);

-- Recent configuration changes
CREATE VIEW config.recent_configuration_changes AS
SELECT 
    cc.change_id,
    cc.change_type,
    cc.table_name,
    cc.change_summary,
    cc.changed_by,
    cc.approval_status,
    cc.deployed,
    cc.created_at
FROM config.configuration_changes cc
WHERE cc.created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY cc.created_at DESC;

GRANT SELECT ON config.active_models_by_service TO facesocial_user;
GRANT SELECT ON config.current_environment_settings TO facesocial_user;
GRANT SELECT ON config.active_feature_flags TO facesocial_user;
GRANT SELECT ON config.recent_configuration_changes TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Configuration and Settings tables created successfully!';
    RAISE NOTICE 'Tables: ai_models, service_settings, feature_flags, api_rate_limits, system_parameters, configuration_templates, environment_configs, configuration_changes';
    RAISE NOTICE 'Views: active_models_by_service, current_environment_settings, active_feature_flags, recent_configuration_changes';
    RAISE NOTICE 'Default configurations inserted for environments, system parameters, and rate limits';
    RAISE NOTICE 'Configuration change logging and audit trail enabled';
END $$;