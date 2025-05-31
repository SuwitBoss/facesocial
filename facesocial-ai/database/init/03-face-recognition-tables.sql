-- Face Recognition Service Tables
-- Handles face registration, verification, identification and embeddings

-- Face data table - ข้อมูลใบหน้าที่ลงทะเบียน
CREATE TABLE face_recognition.face_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    face_id VARCHAR(64) NOT NULL UNIQUE, -- External face ID for Milvus
    
    -- Face metadata
    image_hash VARCHAR(64) NOT NULL REFERENCES core.image_metadata(image_hash),
    face_quality_score DECIMAL(5,4), -- 0.0000-1.0000
    face_landmarks JSONB, -- Facial landmark points
    face_attributes JSONB, -- Additional face attributes
    
    -- Bounding box information
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    bbox_confidence DECIMAL(5,4),
    
    -- Embedding information
    embedding_version VARCHAR(20) NOT NULL, -- Model version used
    embedding_dimension INTEGER NOT NULL, -- 256 or 512
    milvus_collection VARCHAR(50) NOT NULL, -- Collection name in Milvus
    embedding_hash VARCHAR(64), -- Hash of the embedding vector
    
    -- Registration context
    registration_source VARCHAR(50), -- 'signup', 'mobile_app', 'web', etc.
    registration_device_info JSONB,
    ip_address INET,
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'suspended', 'deleted'
    verification_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Face quality metrics
    pose_yaw DECIMAL(7,4), -- Head rotation angles
    pose_pitch DECIMAL(7,4),
    pose_roll DECIMAL(7,4),
    eye_distance DECIMAL(8,4), -- Distance between eyes in pixels
    face_area INTEGER, -- Face area in pixels
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Face recognition sessions - เซสชันการตรวจสอบใบหน้า
CREATE TABLE face_recognition.recognition_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Recognition request
    operation_type VARCHAR(20) NOT NULL, -- 'register', 'verify', 'identify'
    input_image_hash VARCHAR(64) NOT NULL,
    input_face_bbox JSONB, -- Bounding box of input face
    
    -- Results
    matched_face_id UUID REFERENCES face_recognition.face_data(id),
    confidence_score DECIMAL(5,4),
    similarity_score DECIMAL(5,4),
    threshold_used DECIMAL(5,4), -- Threshold used for matching
    recognition_result VARCHAR(20), -- 'success', 'failed', 'rejected'
    failure_reason VARCHAR(100), -- Reason for failure if any
    
    -- Processing details
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    preprocessing_steps JSONB,
    distance_metric VARCHAR(20), -- 'cosine', 'euclidean', 'manhattan'
    
    -- Multiple candidates (for identification)
    candidate_matches JSONB, -- Array of {face_id, similarity_score}
    
    -- Context
    request_source VARCHAR(50), -- 'login', 'post_tagging', 'video_call', etc.
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Security and anti-fraud
    risk_score DECIMAL(5,4), -- Risk assessment score
    fraud_indicators JSONB, -- Detected fraud patterns
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Face groups - การจัดกลุ่มใบหน้า
CREATE TABLE face_recognition.face_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_name VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    description TEXT,
    face_count INTEGER DEFAULT 0,
    privacy_level VARCHAR(20) DEFAULT 'private', -- 'public', 'friends', 'private'
    
    -- Group settings
    auto_add_enabled BOOLEAN DEFAULT FALSE, -- Auto-add similar faces
    similarity_threshold DECIMAL(5,4) DEFAULT 0.8, -- Threshold for auto-add
    
    -- Group metadata
    group_type VARCHAR(30) DEFAULT 'manual', -- 'manual', 'auto_family', 'auto_friends'
    tags JSONB, -- Array of tags
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Face group members - สมาชิกในกลุ่มใบหน้า
CREATE TABLE face_recognition.face_group_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id UUID NOT NULL REFERENCES face_recognition.face_groups(id),
    face_id UUID NOT NULL REFERENCES face_recognition.face_data(id),
    added_by UUID REFERENCES core.users(user_id),
    
    -- Member metadata
    member_role VARCHAR(20) DEFAULT 'member', -- 'admin', 'member'
    is_representative BOOLEAN DEFAULT FALSE, -- Is this the main face for the person
    confidence_score DECIMAL(5,4), -- How confident we are this belongs to group
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(group_id, face_id)
);

-- Face recognition models configuration
CREATE TABLE face_recognition.recognition_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'arcface', 'adaface', 'facenet'
    
    -- Model file information
    model_path TEXT NOT NULL,
    model_size_mb DECIMAL(10,2),
    embedding_dimension INTEGER NOT NULL,
    
    -- Performance characteristics
    accuracy_lfw DECIMAL(5,4), -- Accuracy on LFW dataset
    accuracy_cfp DECIMAL(5,4), -- Accuracy on CFP dataset
    inference_time_ms DECIMAL(10,2), -- Average inference time
    
    -- Configuration
    input_size JSONB, -- {width, height}
    preprocessing_config JSONB,
    normalization_method VARCHAR(30), -- 'l2', 'l1', 'none'
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    description TEXT,
    training_dataset TEXT,
    training_date DATE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Face similarity cache - แคชผลการเปรียบเทียบใบหน้า
CREATE TABLE face_recognition.similarity_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    face_id_1 UUID NOT NULL REFERENCES face_recognition.face_data(id),
    face_id_2 UUID NOT NULL REFERENCES face_recognition.face_data(id),
    
    -- Similarity metrics
    similarity_score DECIMAL(5,4) NOT NULL,
    distance_metric VARCHAR(20) NOT NULL, -- 'cosine', 'euclidean'
    model_version VARCHAR(20) NOT NULL,
    
    -- Cache metadata
    computation_time_ms INTEGER,
    cache_hits INTEGER DEFAULT 0,
    
    -- Timestamps
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
    
    UNIQUE(face_id_1, face_id_2, model_version),
    
    -- Ensure face_id_1 < face_id_2 for consistency
    CHECK (face_id_1 < face_id_2)
);

-- Face templates - เทมเพลตใบหน้าสำหรับการจดจำที่รวดเร็ว
CREATE TABLE face_recognition.face_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    template_name VARCHAR(100),
    
    -- Template data
    primary_face_id UUID REFERENCES face_recognition.face_data(id),
    face_ids JSONB NOT NULL, -- Array of face IDs in this template
    template_embedding BYTEA, -- Averaged/optimized embedding
    
    -- Template quality
    template_quality_score DECIMAL(5,4),
    face_count INTEGER DEFAULT 1,
    consistency_score DECIMAL(5,4), -- How consistent faces are in template
    
    -- Usage statistics
    verification_count INTEGER DEFAULT 0,
    identification_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4),
    
    -- Template configuration
    embedding_method VARCHAR(30), -- 'average', 'weighted_average', 'best_quality'
    quality_threshold DECIMAL(5,4) DEFAULT 0.7,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',
    is_primary BOOLEAN DEFAULT FALSE, -- Is this the primary template for user
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization

-- Face data indexes
CREATE INDEX idx_face_data_user_id ON face_recognition.face_data(user_id);
CREATE INDEX idx_face_data_face_id ON face_recognition.face_data(face_id);
CREATE INDEX idx_face_data_status ON face_recognition.face_data(status);
CREATE INDEX idx_face_data_created_at ON face_recognition.face_data(created_at);
CREATE INDEX idx_face_data_quality ON face_recognition.face_data(face_quality_score);
CREATE INDEX idx_face_data_embedding_version ON face_recognition.face_data(embedding_version);
CREATE INDEX idx_face_data_milvus_collection ON face_recognition.face_data(milvus_collection);

-- Recognition sessions indexes
CREATE INDEX idx_recognition_sessions_user_id ON face_recognition.recognition_sessions(user_id);
CREATE INDEX idx_recognition_sessions_session_id ON face_recognition.recognition_sessions(session_id);
CREATE INDEX idx_recognition_sessions_operation ON face_recognition.recognition_sessions(operation_type);
CREATE INDEX idx_recognition_sessions_created_at ON face_recognition.recognition_sessions(created_at);
CREATE INDEX idx_recognition_sessions_result ON face_recognition.recognition_sessions(recognition_result);
CREATE INDEX idx_recognition_sessions_matched_face ON face_recognition.recognition_sessions(matched_face_id);

-- Face groups indexes
CREATE INDEX idx_face_groups_user_id ON face_recognition.face_groups(user_id);
CREATE INDEX idx_face_groups_privacy ON face_recognition.face_groups(privacy_level);
CREATE INDEX idx_face_groups_type ON face_recognition.face_groups(group_type);

-- Face group members indexes
CREATE INDEX idx_face_group_members_group_id ON face_recognition.face_group_members(group_id);
CREATE INDEX idx_face_group_members_face_id ON face_recognition.face_group_members(face_id);
CREATE INDEX idx_face_group_members_added_by ON face_recognition.face_group_members(added_by);

-- Recognition models indexes
CREATE INDEX idx_recognition_models_name ON face_recognition.recognition_models(model_name);
CREATE INDEX idx_recognition_models_type ON face_recognition.recognition_models(model_type);
CREATE INDEX idx_recognition_models_status ON face_recognition.recognition_models(status);
CREATE INDEX idx_recognition_models_is_default ON face_recognition.recognition_models(is_default);

-- Similarity cache indexes
CREATE INDEX idx_similarity_cache_face1 ON face_recognition.similarity_cache(face_id_1);
CREATE INDEX idx_similarity_cache_face2 ON face_recognition.similarity_cache(face_id_2);
CREATE INDEX idx_similarity_cache_score ON face_recognition.similarity_cache(similarity_score);
CREATE INDEX idx_similarity_cache_expires_at ON face_recognition.similarity_cache(expires_at);

-- Face templates indexes
CREATE INDEX idx_face_templates_user_id ON face_recognition.face_templates(user_id);
CREATE INDEX idx_face_templates_primary_face ON face_recognition.face_templates(primary_face_id);
CREATE INDEX idx_face_templates_status ON face_recognition.face_templates(status);
CREATE INDEX idx_face_templates_is_primary ON face_recognition.face_templates(is_primary);

-- Create triggers for updated_at columns
CREATE TRIGGER update_face_data_updated_at BEFORE UPDATE ON face_recognition.face_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_groups_updated_at BEFORE UPDATE ON face_recognition.face_groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recognition_models_updated_at BEFORE UPDATE ON face_recognition.recognition_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_templates_updated_at BEFORE UPDATE ON face_recognition.face_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA face_recognition TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA face_recognition TO facesocial_user;

-- Create views for common queries

-- Active faces with quality metrics
CREATE VIEW face_recognition.active_faces AS
SELECT 
    fd.*,
    im.width as image_width,
    im.height as image_height,
    im.format as image_format,
    u.user_id
FROM face_recognition.face_data fd
JOIN core.image_metadata im ON fd.image_hash = im.image_hash
JOIN core.users u ON fd.user_id = u.user_id
WHERE fd.status = 'active'
AND fd.deleted_at IS NULL;

-- User face statistics
CREATE VIEW face_recognition.user_face_stats AS
SELECT 
    u.user_id,
    COUNT(fd.id) as total_faces,
    AVG(fd.face_quality_score) as avg_quality_score,
    MAX(fd.face_quality_score) as best_quality_score,
    COUNT(CASE WHEN fd.face_quality_score > 0.8 THEN 1 END) as high_quality_faces,
    MAX(fd.last_used_at) as last_face_used,
    MIN(fd.created_at) as first_face_registered
FROM core.users u
LEFT JOIN face_recognition.face_data fd ON u.user_id = fd.user_id 
WHERE fd.status = 'active' OR fd.status IS NULL
GROUP BY u.user_id;

-- Recognition performance summary
CREATE VIEW face_recognition.recognition_performance AS
SELECT 
    DATE(rs.created_at) as date,
    rs.operation_type,
    rs.model_version,
    COUNT(*) as total_attempts,
    COUNT(CASE WHEN rs.recognition_result = 'success' THEN 1 END) as successful_attempts,
    AVG(rs.confidence_score) as avg_confidence,
    AVG(rs.processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN rs.recognition_result = 'success' THEN 1 END)::DECIMAL / COUNT(*) as success_rate
FROM face_recognition.recognition_sessions rs
WHERE rs.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(rs.created_at), rs.operation_type, rs.model_version
ORDER BY date DESC, rs.operation_type;

GRANT SELECT ON face_recognition.active_faces TO facesocial_user;
GRANT SELECT ON face_recognition.user_face_stats TO facesocial_user;
GRANT SELECT ON face_recognition.recognition_performance TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Face Recognition tables created successfully!';
    RAISE NOTICE 'Tables: face_data, recognition_sessions, face_groups, face_group_members, recognition_models, similarity_cache, face_templates';
    RAISE NOTICE 'Views: active_faces, user_face_stats, recognition_performance';
    RAISE NOTICE 'Indexes and triggers configured for optimal performance';
END $$;