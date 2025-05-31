-- Face Detection Service Tables
-- Handles multi-face detection, landmarks, and quality assessment

-- Detection requests table
CREATE TABLE face_detection.detection_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input data
    image_hash VARCHAR(64) NOT NULL REFERENCES core.image_metadata(image_hash),
    image_url TEXT,
    image_metadata JSONB, -- resolution, format, etc.
    
    -- Detection parameters
    min_face_size INTEGER DEFAULT 40,
    max_faces INTEGER DEFAULT 100,
    detection_confidence DECIMAL(5,4) DEFAULT 0.7,
    return_landmarks BOOLEAN DEFAULT TRUE,
    return_attributes BOOLEAN DEFAULT FALSE,
    face_alignment BOOLEAN DEFAULT FALSE,
    crop_faces BOOLEAN DEFAULT FALSE,
    enhance_quality BOOLEAN DEFAULT FALSE,
    
    -- Results summary
    faces_detected INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    
    -- Detection statistics
    detection_stats JSONB DEFAULT '{}', -- Additional detection metrics
    
    -- Context
    request_source VARCHAR(50), -- 'post_upload', 'profile_photo', 'auto_tag', etc.
    session_id VARCHAR(128) REFERENCES core.sessions(session_id),
    ip_address INET,
    
    -- Status
    status VARCHAR(20) DEFAULT 'processing', -- 'processing', 'completed', 'failed'
    error_message TEXT,
    error_code VARCHAR(50),
    
    -- Processing details
    queue_position INTEGER,
    processing_node VARCHAR(50), -- Which server/container processed this
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Detected faces table
CREATE TABLE face_detection.detected_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_request_id UUID NOT NULL REFERENCES face_detection.detection_requests(id),
    face_index INTEGER NOT NULL, -- Face order in image (0, 1, 2, ...)
    face_id VARCHAR(32) NOT NULL DEFAULT generate_short_uuid(), -- Unique ID for this detected face
    
    -- Bounding box
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    
    -- Detection confidence
    detection_confidence DECIMAL(5,4) NOT NULL,
    face_quality_score DECIMAL(5,4),
    
    -- Face orientation and pose
    face_angle JSONB, -- {yaw, pitch, roll}
    face_pose_score DECIMAL(5,4),
    
    -- Facial landmarks (5-point, 68-point, or 106-point)
    landmarks JSONB, -- Array of {x, y} coordinates
    landmarks_type VARCHAR(10), -- '5_point', '68_point', '106_point'
    landmarks_confidence DECIMAL(5,4),
    
    -- Face quality metrics
    quality_metrics JSONB DEFAULT '{}', -- {sharpness, brightness, contrast, blur, etc.}
    
    -- Face attributes (basic analysis)
    estimated_age DECIMAL(5,2),
    estimated_gender VARCHAR(10),
    gender_confidence DECIMAL(5,4),
    emotions JSONB, -- {happy, sad, angry, surprised, neutral, etc.}
    dominant_emotion VARCHAR(20),
    emotion_confidence DECIMAL(5,4),
    
    -- Occlusion detection
    occlusion_analysis JSONB, -- {left_eye, right_eye, nose, mouth, etc.}
    occlusion_score DECIMAL(5,4), -- Overall occlusion level
    
    -- Face region crops (if requested)
    aligned_face_data BYTEA, -- Base64 encoded aligned face image
    cropped_face_data BYTEA, -- Base64 encoded cropped face image
    face_crop_coords JSONB, -- Coordinates used for cropping
    
    -- Recognition readiness
    recognition_ready BOOLEAN DEFAULT FALSE, -- Whether face is suitable for recognition
    recognition_quality_score DECIMAL(5,4), -- Score for recognition quality
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_versions JSONB, -- {detector: 'v1.0', landmarks: 'v2.0', etc.}
    
    -- Recognition results (if performed in same request)
    recognized_user_id UUID REFERENCES core.users(user_id),
    recognition_confidence DECIMAL(5,4),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_face_per_request UNIQUE(detection_request_id, face_index)
);

-- Face tracking sessions (for video/stream processing)
CREATE TABLE face_detection.tracking_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Tracking configuration
    tracking_config JSONB NOT NULL, -- Frame rate, detection interval, etc.
    max_faces INTEGER DEFAULT 10,
    tracking_confidence DECIMAL(5,4) DEFAULT 0.8,
    
    -- Session status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'paused', 'stopped', 'completed'
    
    -- Statistics
    total_frames_processed INTEGER DEFAULT 0,
    total_faces_tracked INTEGER DEFAULT 0,
    unique_faces_count INTEGER DEFAULT 0,
    
    -- Current state
    active_tracks JSONB DEFAULT '[]', -- Currently tracked faces
    last_frame_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Callbacks and notifications
    webhook_url TEXT,
    notification_settings JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Face tracks (individual face tracking across frames)
CREATE TABLE face_detection.face_tracks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracking_session_id UUID NOT NULL REFERENCES face_detection.tracking_sessions(id),
    track_id VARCHAR(32) NOT NULL, -- Unique ID for this track within session
    
    -- Track metadata
    first_detection_id UUID REFERENCES face_detection.detected_faces(id),
    last_detection_id UUID REFERENCES face_detection.detected_faces(id),
    
    -- Track statistics
    detection_count INTEGER DEFAULT 1,
    track_confidence DECIMAL(5,4),
    track_quality_score DECIMAL(5,4),
    
    -- Track duration
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Recognition information (if available)
    identified_user_id UUID REFERENCES core.users(user_id),
    identification_confidence DECIMAL(5,4),
    
    -- Track status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'lost', 'merged', 'completed'
    
    CONSTRAINT unique_track_per_session UNIQUE(tracking_session_id, track_id)
);

-- Face detection models configuration
CREATE TABLE face_detection.model_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'face_detector', 'landmark_detector', 'quality_assessor'
    
    -- Model file information
    model_path TEXT NOT NULL,
    model_size_mb DECIMAL(10,2),
    model_format VARCHAR(20), -- 'onnx', 'pytorch', 'tensorflow'
    
    -- Model performance characteristics
    input_size JSONB, -- {width, height, channels}
    output_format JSONB, -- Description of model outputs
    avg_inference_time_ms DECIMAL(10,2),
    accuracy_metrics JSONB,
    
    -- Hardware requirements
    min_gpu_memory_mb INTEGER,
    supports_batch_processing BOOLEAN DEFAULT FALSE,
    max_batch_size INTEGER DEFAULT 1,
    
    -- Model status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    description TEXT,
    training_dataset TEXT,
    training_date DATE,
    
    -- Configuration parameters
    config_params JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_model_version UNIQUE(model_name, model_version)
);

-- Batch processing jobs
CREATE TABLE face_detection.batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Job configuration
    job_name VARCHAR(100),
    job_description TEXT,
    detection_config JSONB NOT NULL,
    
    -- Input data
    total_images INTEGER NOT NULL,
    input_manifest JSONB, -- List of images to process
    
    -- Processing status
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'cancelled'
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    
    -- Results summary
    images_processed INTEGER DEFAULT 0,
    images_failed INTEGER DEFAULT 0,
    total_faces_detected INTEGER DEFAULT 0,
    
    -- Processing details
    processing_priority INTEGER DEFAULT 5, -- 1-10, higher = more urgent
    max_concurrent_workers INTEGER DEFAULT 4,
    estimated_completion_time TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Callbacks
    callback_url TEXT,
    notification_email TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance optimization

-- Detection requests indexes
CREATE INDEX idx_detection_requests_request_id ON face_detection.detection_requests(request_id);
CREATE INDEX idx_detection_requests_user_id ON face_detection.detection_requests(user_id);
CREATE INDEX idx_detection_requests_status ON face_detection.detection_requests(status);
CREATE INDEX idx_detection_requests_created_at ON face_detection.detection_requests(created_at);
CREATE INDEX idx_detection_requests_image_hash ON face_detection.detection_requests(image_hash);
CREATE INDEX idx_detection_requests_source ON face_detection.detection_requests(request_source);

-- Detected faces indexes
CREATE INDEX idx_detected_faces_request_id ON face_detection.detected_faces(detection_request_id);
CREATE INDEX idx_detected_faces_face_id ON face_detection.detected_faces(face_id);
CREATE INDEX idx_detected_faces_recognized_user ON face_detection.detected_faces(recognized_user_id);
CREATE INDEX idx_detected_faces_quality ON face_detection.detected_faces(face_quality_score);
CREATE INDEX idx_detected_faces_recognition_ready ON face_detection.detected_faces(recognition_ready);

-- Tracking sessions indexes
CREATE INDEX idx_tracking_sessions_session_id ON face_detection.tracking_sessions(session_id);
CREATE INDEX idx_tracking_sessions_user_id ON face_detection.tracking_sessions(user_id);
CREATE INDEX idx_tracking_sessions_status ON face_detection.tracking_sessions(status);
CREATE INDEX idx_tracking_sessions_created_at ON face_detection.tracking_sessions(created_at);

-- Face tracks indexes
CREATE INDEX idx_face_tracks_tracking_session ON face_detection.face_tracks(tracking_session_id);
CREATE INDEX idx_face_tracks_track_id ON face_detection.face_tracks(track_id);
CREATE INDEX idx_face_tracks_identified_user ON face_detection.face_tracks(identified_user_id);
CREATE INDEX idx_face_tracks_status ON face_detection.face_tracks(status);

-- Model configs indexes
CREATE INDEX idx_model_configs_name ON face_detection.model_configs(model_name);
CREATE INDEX idx_model_configs_type ON face_detection.model_configs(model_type);
CREATE INDEX idx_model_configs_status ON face_detection.model_configs(status);
CREATE INDEX idx_model_configs_is_default ON face_detection.model_configs(is_default);

-- Batch jobs indexes
CREATE INDEX idx_batch_jobs_batch_id ON face_detection.batch_jobs(batch_id);
CREATE INDEX idx_batch_jobs_user_id ON face_detection.batch_jobs(user_id);
CREATE INDEX idx_batch_jobs_status ON face_detection.batch_jobs(status);
CREATE INDEX idx_batch_jobs_created_at ON face_detection.batch_jobs(created_at);

-- Create triggers for updated_at columns
CREATE TRIGGER update_tracking_sessions_updated_at BEFORE UPDATE ON face_detection.tracking_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at BEFORE UPDATE ON face_detection.model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA face_detection TO facesocial_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA face_detection TO facesocial_user;

-- Create views for common queries

-- Active detection requests with summary
CREATE VIEW face_detection.active_detections AS
SELECT 
    dr.request_id,
    dr.user_id,
    dr.faces_detected,
    dr.status,
    dr.processing_time_ms,
    dr.created_at,
    COUNT(df.id) as detected_faces_count
FROM face_detection.detection_requests dr
LEFT JOIN face_detection.detected_faces df ON dr.id = df.detection_request_id
WHERE dr.status IN ('processing', 'completed')
GROUP BY dr.id;

-- High quality faces ready for recognition
CREATE VIEW face_detection.recognition_ready_faces AS
SELECT 
    df.*,
    dr.user_id,
    dr.image_hash
FROM face_detection.detected_faces df
JOIN face_detection.detection_requests dr ON df.detection_request_id = dr.id
WHERE df.recognition_ready = TRUE
AND df.face_quality_score > 0.7
AND dr.status = 'completed';

GRANT SELECT ON face_detection.active_detections TO facesocial_user;
GRANT SELECT ON face_detection.recognition_ready_faces TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Face Detection tables created successfully!';
    RAISE NOTICE 'Tables: detection_requests, detected_faces, tracking_sessions, face_tracks, model_configs, batch_jobs';
    RAISE NOTICE 'Views: active_detections, recognition_ready_faces';
    RAISE NOTICE 'Indexes and triggers configured for optimal performance';
END $$;