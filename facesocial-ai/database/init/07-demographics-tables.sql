-- Demographics Detection Service Tables
-- Handles age and gender detection, and other demographic analysis

-- Demographic analyses - การวิเคราะห์ demographics
CREATE TABLE demographics.demographic_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input data
    image_hash VARCHAR(64) NOT NULL REFERENCES core.image_metadata(image_hash),
    face_detection_id UUID, -- Reference to face_detection if available
    
    -- Analysis configuration
    analysis_types JSONB NOT NULL, -- ['age', 'gender', 'emotion', 'ethnicity', 'attractiveness']
    include_confidence_intervals BOOLEAN DEFAULT TRUE,
    include_probability_distribution BOOLEAN DEFAULT FALSE,
    
    -- Processing details
    faces_analyzed INTEGER DEFAULT 0,
    model_versions JSONB, -- {age: 'v1.0', gender: 'v2.1', emotion: 'v1.5'}
    processing_time_ms INTEGER,
    
    -- Quality metrics
    overall_analysis_quality DECIMAL(5,4),
    face_quality_threshold DECIMAL(5,4) DEFAULT 0.5,
    
    -- Context
    request_source VARCHAR(50), -- 'content_analysis', 'user_profiling', 'targeting', 'moderation'
    batch_id VARCHAR(64), -- For batch processing
    analysis_purpose VARCHAR(50), -- 'analytics', 'personalization', 'safety', 'research'
    
    -- Privacy and compliance
    anonymized BOOLEAN DEFAULT TRUE,
    data_retention_days INTEGER DEFAULT 90,
    consent_obtained BOOLEAN DEFAULT FALSE,
    gdpr_compliant BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Face demographics - ผลการวิเคราะห์รายใบหน้า
CREATE TABLE demographics.face_demographics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES demographics.demographic_analyses(id),
    face_index INTEGER NOT NULL,
    face_id VARCHAR(32), -- Unique identifier for this face in the image
    
    -- Face location (if not from face_detection)
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    face_quality_score DECIMAL(5,4),
    
    -- Age estimation
    estimated_age DECIMAL(5,2) NOT NULL,
    age_confidence DECIMAL(5,4),
    age_range_min INTEGER,
    age_range_max INTEGER,
    age_group VARCHAR(20), -- 'child', 'teenager', 'young_adult', 'adult', 'middle_aged', 'senior'
    age_probability_distribution JSONB, -- Age probability for each year/range
    
    -- Gender estimation
    estimated_gender VARCHAR(10) NOT NULL, -- 'male', 'female', 'non_binary', 'unknown'
    gender_confidence DECIMAL(5,4),
    gender_probability JSONB, -- {male: 0.75, female: 0.25}
    
    -- Emotion detection
    primary_emotion VARCHAR(20), -- 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'neutral'
    emotion_confidence DECIMAL(5,4),
    emotion_scores JSONB, -- Scores for all emotions
    emotion_intensity DECIMAL(5,4), -- How intense the emotion is
    
    -- Ethnicity estimation (optional, privacy-sensitive)
    estimated_ethnicity VARCHAR(30), -- 'asian', 'african', 'caucasian', 'hispanic', 'middle_eastern', 'mixed'
    ethnicity_confidence DECIMAL(5,4),
    ethnicity_probabilities JSONB, -- Probability distribution
    
    -- Additional attributes
    attractiveness_score DECIMAL(5,4), -- Subjective attractiveness measure
    facial_hair VARCHAR(20), -- 'none', 'beard', 'mustache', 'goatee', 'stubble'
    facial_hair_confidence DECIMAL(5,4),
    
    hair_color VARCHAR(20), -- 'black', 'brown', 'blonde', 'red', 'gray', 'white', 'other'
    hair_color_confidence DECIMAL(5,4),
    
    eye_color VARCHAR(20), -- 'brown', 'blue', 'green', 'hazel', 'gray', 'amber'
    eye_color_confidence DECIMAL(5,4),
    
    glasses VARCHAR(20), -- 'none', 'prescription', 'sunglasses', 'reading'
    glasses_confidence DECIMAL(5,4),
    
    makeup_present BOOLEAN,
    makeup_intensity DECIMAL(5,4),
    
    -- Facial structure analysis
    face_shape VARCHAR(20), -- 'oval', 'round', 'square', 'heart', 'diamond', 'oblong'
    jawline_strength DECIMAL(5,4),
    cheekbone_prominence DECIMAL(5,4),
    
    -- Expression analysis
    smile_intensity DECIMAL(5,4),
    eye_openness DECIMAL(5,4),
    mouth_openness DECIMAL(5,4),
    
    -- Pose and orientation
    head_pose JSONB, -- {yaw, pitch, roll}
    gaze_direction JSONB, -- {horizontal, vertical}
    
    -- Quality and reliability indicators
    detection_quality DECIMAL(5,4),
    analysis_reliability DECIMAL(5,4),
    uncertainty_score DECIMAL(5,4),
    
    -- Model-specific results
    model_predictions JSONB, -- Raw predictions from different models
    ensemble_weights JSONB, -- Weights used in ensemble prediction
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(analysis_id, face_index)
);

-- Demographic models - โมเดลสำหรับการวิเคราะห์
CREATE TABLE demographics.demographic_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'age_estimation', 'gender_classification', 'emotion_recognition', 'ethnicity_classification'
    
    -- Model architecture
    architecture VARCHAR(50), -- 'resnet', 'efficientnet', 'mobilenet', 'vit'
    model_size_mb DECIMAL(10,2),
    input_size JSONB, -- {width, height, channels}
    
    -- Performance metrics by category
    overall_accuracy DECIMAL(5,4),
    
    -- Age-specific metrics
    age_mae DECIMAL(5,2), -- Mean Absolute Error for age
    age_accuracy_5yr DECIMAL(5,4), -- Accuracy within 5 years
    age_accuracy_10yr DECIMAL(5,4), -- Accuracy within 10 years
    
    -- Gender-specific metrics
    gender_accuracy DECIMAL(5,4),
    gender_precision DECIMAL(5,4),
    gender_recall DECIMAL(5,4),
    gender_f1_score DECIMAL(5,4),
    
    -- Emotion-specific metrics
    emotion_accuracy DECIMAL(5,4),
    emotion_confusion_matrix JSONB,
    
    -- Bias and fairness metrics
    age_bias_score DECIMAL(5,4), -- Bias across age groups
    gender_bias_score DECIMAL(5,4), -- Bias across genders
    ethnicity_bias_score DECIMAL(5,4), -- Bias across ethnicities
    fairness_metrics JSONB, -- Various fairness measurements
    
    -- Dataset information
    training_dataset TEXT,
    training_size INTEGER,
    validation_dataset TEXT,
    test_dataset TEXT,
    dataset_diversity_score DECIMAL(5,4),
    
    -- Processing performance
    inference_time_ms DECIMAL(10,2),
    batch_processing_capability BOOLEAN DEFAULT TRUE,
    optimal_batch_size INTEGER DEFAULT 8,
    
    -- Hardware requirements
    min_gpu_memory_mb INTEGER,
    supports_cpu BOOLEAN DEFAULT TRUE,
    supports_mobile BOOLEAN DEFAULT FALSE,
    
    -- Configuration
    preprocessing_config JSONB,
    augmentation_config JSONB,
    postprocessing_config JSONB,
    threshold_config JSONB,
    
    -- Ethical considerations
    bias_mitigation_applied BOOLEAN DEFAULT FALSE,
    fairness_constraints JSONB,
    ethical_guidelines_followed TEXT,
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    is_default BOOLEAN DEFAULT FALSE,
    deployment_date DATE,
    deprecation_date DATE,
    
    -- Metadata
    description TEXT,
    paper_reference TEXT,
    license VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version, model_type)
);

-- Age groups configuration - การจัดกลุ่มอายุ
CREATE TABLE demographics.age_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_name VARCHAR(30) NOT NULL UNIQUE,
    min_age INTEGER NOT NULL,
    max_age INTEGER NOT NULL,
    
    -- Display information
    display_name VARCHAR(50),
    description TEXT,
    icon VARCHAR(50),
    color_code VARCHAR(7), -- Hex color code
    
    -- Business/legal considerations
    requires_parental_consent BOOLEAN DEFAULT FALSE,
    content_restrictions JSONB, -- Content restrictions for this age group
    privacy_level VARCHAR(20) DEFAULT 'standard', -- 'minimal', 'standard', 'enhanced'
    
    -- Cultural variations
    cultural_variations JSONB, -- Different age group definitions by culture
    
    -- Usage statistics
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    is_active BOOLEAN DEFAULT TRUE,
    sort_order INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (min_age <= max_age)
);

-- Demographic insights - ข้อมูลเชิงลึกทางประชากรศาสตร์
CREATE TABLE demographics.demographic_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_date DATE NOT NULL,
    
    -- Aggregation scope
    scope_type VARCHAR(30) NOT NULL, -- 'global', 'user', 'content_type', 'geographic'
    scope_value VARCHAR(100), -- Specific value for the scope
    
    -- Age insights
    avg_age DECIMAL(5,2),
    median_age DECIMAL(5,2),
    age_distribution JSONB, -- Distribution across age groups
    age_variance DECIMAL(8,4),
    
    -- Gender insights
    gender_distribution JSONB, -- Distribution of genders
    gender_balance_score DECIMAL(5,4), -- How balanced the gender distribution is
    
    -- Emotion insights
    dominant_emotion VARCHAR(20),
    emotion_diversity_score DECIMAL(5,4), -- How diverse emotions are
    emotion_distribution JSONB,
    positive_emotion_ratio DECIMAL(5,4),
    
    -- Temporal patterns
    hour_of_day_patterns JSONB, -- Patterns by hour
    day_of_week_patterns JSONB, -- Patterns by day
    seasonal_patterns JSONB, -- Seasonal variations
    
    -- Quality metrics
    analysis_count INTEGER NOT NULL,
    confidence_score DECIMAL(5,4),
    data_quality_score DECIMAL(5,4),
    
    -- Trends
    trend_direction VARCHAR(20), -- 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength DECIMAL(5,4),
    compared_to_previous_period JSONB, -- Comparison with previous period
    
    -- Statistical significance
    sample_size INTEGER,
    confidence_interval JSONB,
    statistical_significance DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(insight_date, scope_type, scope_value)
);

-- Create indexes for performance optimization

-- Demographic analyses indexes
CREATE INDEX idx_demographic_analyses_user_id ON demographics.demographic_analyses(user_id);
CREATE INDEX idx_demographic_analyses_request_id ON demographics.demographic_analyses(request_id);
CREATE INDEX idx_demographic_analyses_created_at ON demographics.demographic_analyses(created_at);
CREATE INDEX idx_demographic_analyses_batch_id ON demographics.demographic_analyses(batch_id);
CREATE INDEX idx_demographic_analyses_source ON demographics.demographic_analyses(request_source);

-- Face demographics indexes
CREATE INDEX idx_face_demographics_analysis_id ON demographics.face_demographics(analysis_id);
CREATE INDEX idx_face_demographics_age_group ON demographics.face_demographics(age_group);
CREATE INDEX idx_face_demographics_gender ON demographics.face_demographics(estimated_gender);
CREATE INDEX idx_face_demographics_emotion ON demographics.face_demographics(primary_emotion);
CREATE INDEX idx_face_demographics_age ON demographics.face_demographics(estimated_age);
CREATE INDEX idx_face_demographics_quality ON demographics.face_demographics(face_quality_score);

-- Demographic models indexes
CREATE INDEX idx_demographic_models_name ON demographics.demographic_models(model_name);
CREATE INDEX idx_demographic_models_type ON demographics.demographic_models(model_type);
CREATE INDEX idx_demographic_models_status ON demographics.demographic_models(status);
CREATE INDEX idx_demographic_models_is_default ON demographics.demographic_models(is_default);
CREATE INDEX idx_demographic_models_accuracy ON demographics.demographic_models(overall_accuracy);

-- Age groups indexes
CREATE INDEX idx_age_groups_name ON demographics.age_groups(group_name);
CREATE INDEX idx_age_groups_age_range ON demographics.age_groups(min_age, max_age);
CREATE INDEX idx_age_groups_is_active ON demographics.age_groups(is_active);

-- Demographic insights indexes
CREATE INDEX idx_demographic_insights_date ON demographics.demographic_insights(insight_date);
CREATE INDEX idx_demographic_insights_scope ON demographics.demographic_insights(scope_type, scope_value);
CREATE INDEX idx_demographic_insights_created_at ON demographics.demographic_insights(created_at);

-- Create triggers for updated_at columns
CREATE TRIGGER update_demographic_models_updated_at BEFORE UPDATE ON demographics.demographic_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_age_groups_updated_at BEFORE UPDATE ON demographics.age_groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA demographics TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA demographics TO facesocial_user;

-- Insert default age groups
INSERT INTO demographics.age_groups (group_name, min_age, max_age, display_name, description, sort_order) VALUES
('child', 0, 12, 'Children', 'Children and young kids', 1),
('teenager', 13, 19, 'Teenagers', 'Adolescents and teenagers', 2),
('young_adult', 20, 29, 'Young Adults', 'Young adults in their twenties', 3),
('adult', 30, 49, 'Adults', 'Adults in their thirties and forties', 4),
('middle_aged', 50, 64, 'Middle-aged', 'Middle-aged adults', 5),
('senior', 65, 120, 'Seniors', 'Senior citizens and elderly', 6);

-- Create views for common queries

-- Recent demographic analysis summary
CREATE VIEW demographics.recent_analysis_summary AS
SELECT 
    DATE(da.created_at) as analysis_date,
    da.request_source,
    COUNT(*) as total_analyses,
    AVG(da.faces_analyzed) as avg_faces_per_analysis,
    AVG(da.processing_time_ms) as avg_processing_time,
    AVG(da.overall_analysis_quality) as avg_quality_score
FROM demographics.demographic_analyses da
WHERE da.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(da.created_at), da.request_source
ORDER BY analysis_date DESC, da.request_source;

-- Age distribution summary
CREATE VIEW demographics.age_distribution_summary AS
SELECT 
    fd.age_group,
    COUNT(*) as face_count,
    AVG(fd.estimated_age) as avg_age,
    AVG(fd.age_confidence) as avg_confidence,
    COUNT(CASE WHEN fd.estimated_gender = 'male' THEN 1 END) as male_count,
    COUNT(CASE WHEN fd.estimated_gender = 'female' THEN 1 END) as female_count
FROM demographics.face_demographics fd
JOIN demographics.demographic_analyses da ON fd.analysis_id = da.id
WHERE da.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY fd.age_group
ORDER BY 
    CASE fd.age_group 
        WHEN 'child' THEN 1
        WHEN 'teenager' THEN 2
        WHEN 'young_adult' THEN 3
        WHEN 'adult' THEN 4
        WHEN 'middle_aged' THEN 5
        WHEN 'senior' THEN 6
        ELSE 7
    END;

-- Emotion analysis summary
CREATE VIEW demographics.emotion_analysis_summary AS
SELECT 
    fd.primary_emotion,
    COUNT(*) as detection_count,
    AVG(fd.emotion_confidence) as avg_confidence,
    AVG(fd.emotion_intensity) as avg_intensity,
    AVG(fd.estimated_age) as avg_age_of_emotion,
    COUNT(CASE WHEN fd.estimated_gender = 'male' THEN 1 END) as male_count,
    COUNT(CASE WHEN fd.estimated_gender = 'female' THEN 1 END) as female_count
FROM demographics.face_demographics fd
JOIN demographics.demographic_analyses da ON fd.analysis_id = da.id
WHERE da.created_at >= CURRENT_DATE - INTERVAL '7 days'
AND fd.primary_emotion IS NOT NULL
GROUP BY fd.primary_emotion
ORDER BY detection_count DESC;

-- Model performance comparison
CREATE VIEW demographics.model_performance_comparison AS
SELECT 
    dm.model_name,
    dm.model_version,
    dm.model_type,
    dm.overall_accuracy,
    dm.inference_time_ms,
    dm.status,
    COUNT(da.id) as usage_count,
    AVG(da.overall_analysis_quality) as real_world_quality
FROM demographics.demographic_models dm
LEFT JOIN demographics.demographic_analyses da ON da.model_versions @> json_build_object(dm.model_type, dm.model_version)::jsonb
WHERE dm.status = 'active'
GROUP BY dm.id, dm.model_name, dm.model_version, dm.model_type, dm.overall_accuracy, dm.inference_time_ms, dm.status
ORDER BY dm.overall_accuracy DESC, dm.inference_time_ms ASC;

GRANT SELECT ON demographics.recent_analysis_summary TO facesocial_user;
GRANT SELECT ON demographics.age_distribution_summary TO facesocial_user;
GRANT SELECT ON demographics.emotion_analysis_summary TO facesocial_user;
GRANT SELECT ON demographics.model_performance_comparison TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Demographics tables created successfully!';
    RAISE NOTICE 'Tables: demographic_analyses, face_demographics, demographic_models, age_groups, demographic_insights';
    RAISE NOTICE 'Views: recent_analysis_summary, age_distribution_summary, emotion_analysis_summary, model_performance_comparison';
    RAISE NOTICE 'Default age groups inserted: child, teenager, young_adult, adult, middle_aged, senior';
    RAISE NOTICE 'Indexes and triggers configured for optimal performance and analytics';
END $$;