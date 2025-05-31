-- Deepfake Detection Service Tables
-- Handles detection of AI-generated or manipulated content (images and videos)

-- Deepfake analyses - การวิเคราะห์ deepfake
CREATE TABLE deepfake_detection.deepfake_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input data
    content_type VARCHAR(20) NOT NULL, -- 'image', 'video', 'audio'
    content_hash VARCHAR(64) NOT NULL,
    content_url TEXT,
    content_metadata JSONB,
    
    -- Content information
    file_size_bytes BIGINT,
    duration_seconds DECIMAL(10,3), -- For video/audio content
    resolution JSONB, -- {width, height} for images/videos
    frame_rate DECIMAL(6,2), -- For videos
    audio_sample_rate INTEGER, -- For audio content
    
    -- Analysis parameters
    detection_level VARCHAR(20) DEFAULT 'standard', -- 'basic', 'standard', 'advanced', 'forensic'
    analysis_scope JSONB, -- Which aspects to analyze
    priority INTEGER DEFAULT 5, -- 1-10, higher = more urgent
    
    -- Results - Image/Single Frame
    is_deepfake BOOLEAN,
    confidence_score DECIMAL(5,4),
    manipulation_score DECIMAL(5,4), -- How much content has been manipulated
    authenticity_score DECIMAL(5,4), -- How authentic the content appears
    
    -- Results - Video (frame-by-frame analysis)
    total_frames INTEGER,
    analyzed_frames INTEGER,
    deepfake_frame_count INTEGER,
    deepfake_frame_percentage DECIMAL(5,2),
    consistency_score DECIMAL(5,4), -- Temporal consistency across frames
    
    -- Results - Audio (if applicable)
    audio_deepfake_score DECIMAL(5,4),
    voice_consistency_score DECIMAL(5,4),
    
    -- Detailed analysis results
    manipulation_types JSONB, -- ["face_swap", "attribute_manipulation", "expression_transfer", "reenactment"]
    manipulation_regions JSONB, -- Regions where manipulation was detected
    inconsistency_areas JSONB, -- Areas with detected inconsistencies
    temporal_inconsistencies JSONB, -- For videos - frame transition issues
    
    -- Technical analysis
    compression_artifacts JSONB, -- Compression-related artifacts
    frequency_analysis JSONB, -- Frequency domain analysis results
    pixel_level_analysis JSONB, -- Pixel-level inconsistencies
    metadata_analysis JSONB, -- File metadata analysis
    
    -- Face-specific analysis
    face_regions JSONB, -- Detected face regions and their analysis
    facial_landmarks_consistency JSONB, -- Consistency of facial landmarks
    expression_naturalness DECIMAL(5,4), -- How natural facial expressions appear
    
    -- Model predictions
    model_predictions JSONB, -- Predictions from different models
    ensemble_result JSONB, -- Combined result from multiple models
    model_agreement_score DECIMAL(5,4), -- How much models agree
    
    -- Processing details
    model_versions JSONB, -- Versions of models used
    processing_time_ms INTEGER,
    processing_method VARCHAR(30), -- 'sync', 'async', 'batch'
    gpu_processing_time_ms INTEGER,
    cpu_processing_time_ms INTEGER,
    
    -- Job management (for async processing)
    job_id VARCHAR(64),
    job_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'cancelled'
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    worker_node VARCHAR(50), -- Which worker processed this
    
    -- Quality and reliability
    analysis_quality_score DECIMAL(5,4), -- Quality of the analysis itself
    reliability_score DECIMAL(5,4), -- How reliable the results are
    uncertainty_score DECIMAL(5,4), -- Uncertainty in the prediction
    
    -- Context and source
    request_source VARCHAR(50), -- 'content_upload', 'content_moderation', 'verification'
    content_source VARCHAR(50), -- 'user_upload', 'social_media', 'news_media'
    detection_reason VARCHAR(100), -- Why this content was flagged for analysis
    
    -- Error handling
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Notifications and callbacks
    callback_url TEXT,
    notification_settings JSONB,
    webhook_delivered BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Frame analyses - วิเคราะห์รายเฟรมสำหรับวิดีโอ
CREATE TABLE deepfake_detection.frame_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES deepfake_detection.deepfake_analyses(id),
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    
    -- Frame basic info
    frame_hash VARCHAR(64),
    frame_size_bytes INTEGER,
    
    -- Frame analysis results
    is_deepfake BOOLEAN,
    confidence_score DECIMAL(5,4),
    manipulation_types JSONB,
    inconsistent_regions JSONB,
    
    -- Face analysis in frame
    face_count INTEGER DEFAULT 0,
    face_regions JSONB, -- Detected faces and their analysis
    primary_face_analysis JSONB, -- Analysis of the main face
    
    -- Technical analysis
    compression_quality DECIMAL(5,4),
    noise_level DECIMAL(5,4),
    sharpness_score DECIMAL(5,4),
    lighting_consistency DECIMAL(5,4),
    
    -- Temporal analysis (comparing with previous frames)
    motion_consistency DECIMAL(5,4),
    lighting_transition DECIMAL(5,4),
    color_consistency DECIMAL(5,4),
    facial_geometry_consistency DECIMAL(5,4),
    
    -- Model-specific results
    model_predictions JSONB, -- Individual model predictions for this frame
    
    -- Processing info
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(analysis_id, frame_number)
);

-- Detection models - โมเดลสำหรับตรวจจับ deepfake
CREATE TABLE deepfake_detection.detection_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'image_deepfake', 'video_deepfake', 'audio_deepfake', 'multimodal'
    
    -- Model architecture
    architecture VARCHAR(50), -- 'efficientnet', 'xception', 'resnet', 'transformer'
    model_size_mb DECIMAL(10,2),
    input_requirements JSONB, -- Input format, resolution requirements
    
    -- Model capabilities
    supported_content_types JSONB, -- ['image', 'video', 'audio']
    detection_categories JSONB, -- Types of manipulations it can detect
    max_resolution JSONB, -- Maximum supported resolution
    max_duration_seconds INTEGER, -- Maximum video duration
    
    -- Performance metrics
    accuracy_score DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    
    -- Benchmark performance
    celeb_df_accuracy DECIMAL(5,4), -- Performance on CelebDF dataset
    dfdc_accuracy DECIMAL(5,4), -- Performance on DFDC dataset
    faceforensics_accuracy DECIMAL(5,4), -- Performance on FaceForensics++ dataset
    
    -- Processing performance
    avg_processing_time_ms INTEGER,
    frames_per_second DECIMAL(8,2), -- For video processing
    gpu_memory_requirement_mb INTEGER,
    
    -- Model configuration
    preprocessing_config JSONB,
    postprocessing_config JSONB,
    threshold_config JSONB, -- Different thresholds for different scenarios
    ensemble_weight DECIMAL(5,4), -- Weight in ensemble models
    
    -- Training information
    training_dataset TEXT,
    training_date DATE,
    validation_dataset TEXT,
    fine_tuning_info JSONB,
    
    -- Hardware requirements
    min_gpu_memory_mb INTEGER,
    supports_cpu BOOLEAN DEFAULT FALSE,
    supports_batch_processing BOOLEAN DEFAULT TRUE,
    optimal_batch_size INTEGER DEFAULT 1,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing', 'maintenance'
    is_default BOOLEAN DEFAULT FALSE,
    is_ensemble_member BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    description TEXT,
    paper_reference TEXT,
    license_info VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Batch jobs - งานประมวลผลแบบ batch
CREATE TABLE deepfake_detection.batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Job configuration
    job_name VARCHAR(100),
    job_description TEXT,
    detection_config JSONB NOT NULL,
    
    -- Input data
    total_items INTEGER NOT NULL,
    input_manifest JSONB, -- List of content to process
    content_types JSONB, -- Types of content in batch
    
    -- Processing configuration
    processing_priority INTEGER DEFAULT 5,
    max_concurrent_workers INTEGER DEFAULT 4,
    chunk_size INTEGER DEFAULT 10, -- Items per processing chunk
    
    -- Progress tracking
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'cancelled', 'paused'
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    items_skipped INTEGER DEFAULT 0,
    
    -- Results summary
    total_deepfakes_detected INTEGER DEFAULT 0,
    total_authentic_content INTEGER DEFAULT 0,
    avg_confidence_score DECIMAL(5,4),
    high_confidence_detections INTEGER DEFAULT 0, -- Confidence > 0.9
    
    -- Processing details
    estimated_completion_time TIMESTAMP WITH TIME ZONE,
    actual_start_time TIMESTAMP WITH TIME ZONE,
    last_progress_update TIMESTAMP WITH TIME ZONE,
    
    -- Resource usage
    total_processing_time_seconds INTEGER DEFAULT 0,
    total_gpu_time_seconds INTEGER DEFAULT 0,
    peak_memory_usage_mb INTEGER,
    
    -- Error handling
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    retry_policy JSONB, -- Retry configuration
    
    -- Output configuration
    output_format VARCHAR(20) DEFAULT 'json', -- 'json', 'csv', 'xml'
    output_location TEXT,
    include_frame_analysis BOOLEAN DEFAULT FALSE,
    
    -- Notifications
    notification_email TEXT,
    webhook_url TEXT,
    notification_settings JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE
);

-- Content signatures - ลายเซ็นดิจิทัลของเนื้อหา
CREATE TABLE deepfake_detection.content_signatures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    
    -- Content information
    content_type VARCHAR(20) NOT NULL,
    content_size_bytes BIGINT,
    content_metadata JSONB,
    
    -- Perceptual hashing
    perceptual_hash VARCHAR(128), -- Perceptual hash for similarity detection
    dhash VARCHAR(64), -- Difference hash
    phash VARCHAR(64), -- Perceptual hash
    whash VARCHAR(64), -- Wavelet hash
    
    -- Content fingerprinting
    visual_fingerprint JSONB, -- Visual characteristics
    audio_fingerprint JSONB, -- Audio characteristics (if applicable)
    metadata_fingerprint JSONB, -- File metadata characteristics
    
    -- Blockchain/verification
    blockchain_hash VARCHAR(128), -- Blockchain verification hash
    timestamp_verification JSONB, -- Timestamp verification data
    provenance_chain JSONB, -- Content provenance information
    
    -- Analysis cache
    last_analysis_result JSONB, -- Last deepfake analysis result
    last_analysis_date TIMESTAMP WITH TIME ZONE,
    analysis_count INTEGER DEFAULT 0,
    
    -- Usage tracking
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    detection_count INTEGER DEFAULT 0,
    
    -- Classification
    content_category VARCHAR(50), -- 'social_media', 'news', 'entertainment', 'personal'
    risk_level VARCHAR(20), -- 'low', 'medium', 'high'
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- Deepfake analyses indexes
CREATE INDEX idx_deepfake_analyses_user_id ON deepfake_detection.deepfake_analyses(user_id);
CREATE INDEX idx_deepfake_analyses_analysis_id ON deepfake_detection.deepfake_analyses(analysis_id);
CREATE INDEX idx_deepfake_analyses_content_type ON deepfake_detection.deepfake_analyses(content_type);
CREATE INDEX idx_deepfake_analyses_job_status ON deepfake_detection.deepfake_analyses(job_status);
CREATE INDEX idx_deepfake_analyses_is_deepfake ON deepfake_detection.deepfake_analyses(is_deepfake);
CREATE INDEX idx_deepfake_analyses_confidence ON deepfake_detection.deepfake_analyses(confidence_score);
CREATE INDEX idx_deepfake_analyses_created_at ON deepfake_detection.deepfake_analyses(created_at);
CREATE INDEX idx_deepfake_analyses_completed_at ON deepfake_detection.deepfake_analyses(completed_at);
CREATE INDEX idx_deepfake_analyses_priority ON deepfake_detection.deepfake_analyses(priority);

-- Frame analyses indexes
CREATE INDEX idx_frame_analyses_analysis_id ON deepfake_detection.frame_analyses(analysis_id);
CREATE INDEX idx_frame_analyses_frame_number ON deepfake_detection.frame_analyses(frame_number);
CREATE INDEX idx_frame_analyses_timestamp ON deepfake_detection.frame_analyses(timestamp_ms);
CREATE INDEX idx_frame_analyses_is_deepfake ON deepfake_detection.frame_analyses(is_deepfake);
CREATE INDEX idx_frame_analyses_confidence ON deepfake_detection.frame_analyses(confidence_score);

-- Detection models indexes
CREATE INDEX idx_detection_models_name ON deepfake_detection.detection_models(model_name);
CREATE INDEX idx_detection_models_type ON deepfake_detection.detection_models(model_type);
CREATE INDEX idx_detection_models_status ON deepfake_detection.detection_models(status);
CREATE INDEX idx_detection_models_is_default ON deepfake_detection.detection_models(is_default);
CREATE INDEX idx_detection_models_architecture ON deepfake_detection.detection_models(architecture);

-- Batch jobs indexes
CREATE INDEX idx_batch_jobs_batch_id ON deepfake_detection.batch_jobs(batch_id);
CREATE INDEX idx_batch_jobs_user_id ON deepfake_detection.batch_jobs(user_id);
CREATE INDEX idx_batch_jobs_status ON deepfake_detection.batch_jobs(status);
CREATE INDEX idx_batch_jobs_created_at ON deepfake_detection.batch_jobs(created_at);
CREATE INDEX idx_batch_jobs_priority ON deepfake_detection.batch_jobs(processing_priority);

-- Content signatures indexes
CREATE INDEX idx_content_signatures_hash ON deepfake_detection.content_signatures(content_hash);
CREATE INDEX idx_content_signatures_phash ON deepfake_detection.content_signatures(perceptual_hash);
CREATE INDEX idx_content_signatures_content_type ON deepfake_detection.content_signatures(content_type);
CREATE INDEX idx_content_signatures_risk_level ON deepfake_detection.content_signatures(risk_level);
CREATE INDEX idx_content_signatures_last_seen ON deepfake_detection.content_signatures(last_seen_at);

-- Create triggers for updated_at columns
CREATE TRIGGER update_deepfake_analyses_updated_at BEFORE UPDATE ON deepfake_detection.deepfake_analyses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detection_models_updated_at BEFORE UPDATE ON deepfake_detection.detection_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_signatures_updated_at BEFORE UPDATE ON deepfake_detection.content_signatures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA deepfake_detection TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA deepfake_detection TO facesocial_user;

-- Create views for common queries

-- Recent deepfake detections
CREATE VIEW deepfake_detection.recent_detections AS
SELECT 
    da.analysis_id,
    da.user_id,
    da.content_type,
    da.is_deepfake,
    da.confidence_score,
    da.manipulation_types,
    da.processing_time_ms,
    da.job_status,
    da.created_at,
    da.completed_at
FROM deepfake_detection.deepfake_analyses da
WHERE da.created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY da.created_at DESC;

-- Deepfake detection statistics
CREATE VIEW deepfake_detection.detection_stats AS
SELECT 
    DATE(da.created_at) as date,
    da.content_type,
    COUNT(*) as total_analyses,
    COUNT(CASE WHEN da.is_deepfake = true THEN 1 END) as deepfakes_detected,
    COUNT(CASE WHEN da.is_deepfake = false THEN 1 END) as authentic_content,
    AVG(da.confidence_score) as avg_confidence,
    AVG(da.processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN da.confidence_score > 0.9 THEN 1 END) as high_confidence_results
FROM deepfake_detection.deepfake_analyses da
WHERE da.job_status = 'completed'
AND da.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(da.created_at), da.content_type
ORDER BY date DESC, da.content_type;

-- Model performance comparison
CREATE VIEW deepfake_detection.model_performance AS
SELECT 
    dm.model_name,
    dm.model_version,
    dm.model_type,
    dm.accuracy_score,
    dm.f1_score,
    dm.avg_processing_time_ms,
    dm.status,
    COUNT(da.id) as usage_count,
    AVG(da.confidence_score) as avg_real_world_confidence
FROM deepfake_detection.detection_models dm
LEFT JOIN deepfake_detection.deepfake_analyses da ON da.model_versions @> json_build_array(dm.model_version)::jsonb
WHERE dm.status = 'active'
GROUP BY dm.id, dm.model_name, dm.model_version, dm.model_type, dm.accuracy_score, dm.f1_score, dm.avg_processing_time_ms, dm.status
ORDER BY dm.accuracy_score DESC;

-- Active batch jobs summary
CREATE VIEW deepfake_detection.active_batch_jobs AS
SELECT 
    bj.batch_id,
    bj.job_name,
    bj.status,
    bj.progress_percentage,
    bj.items_processed,
    bj.total_items,
    bj.total_deepfakes_detected,
    bj.estimated_completion_time,
    bj.created_at
FROM deepfake_detection.batch_jobs bj
WHERE bj.status IN ('pending', 'processing', 'paused')
ORDER BY bj.processing_priority DESC, bj.created_at ASC;

GRANT SELECT ON deepfake_detection.recent_detections TO facesocial_user;
GRANT SELECT ON deepfake_detection.detection_stats TO facesocial_user;
GRANT SELECT ON deepfake_detection.model_performance TO facesocial_user;
GRANT SELECT ON deepfake_detection.active_batch_jobs TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Deepfake Detection tables created successfully!';
    RAISE NOTICE 'Tables: deepfake_analyses, frame_analyses, detection_models, batch_jobs, content_signatures';
    RAISE NOTICE 'Views: recent_detections, detection_stats, model_performance, active_batch_jobs';
    RAISE NOTICE 'Indexes and triggers configured for optimal performance and security';
END $$;