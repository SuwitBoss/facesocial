-- FaceSocial AI Database Schema Creation
-- This script creates all necessary schemas for AI services

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas for different AI services
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS face_recognition;
CREATE SCHEMA IF NOT EXISTS face_detection;
CREATE SCHEMA IF NOT EXISTS antispoofing;
CREATE SCHEMA IF NOT EXISTS deepfake_detection;
CREATE SCHEMA IF NOT EXISTS demographics;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS config;

-- Grant permissions to application user
GRANT USAGE ON SCHEMA core TO facesocial_user;
GRANT USAGE ON SCHEMA face_recognition TO facesocial_user;
GRANT USAGE ON SCHEMA face_detection TO facesocial_user;
GRANT USAGE ON SCHEMA antispoofing TO facesocial_user;
GRANT USAGE ON SCHEMA deepfake_detection TO facesocial_user;
GRANT USAGE ON SCHEMA demographics TO facesocial_user;
GRANT USAGE ON SCHEMA audit TO facesocial_user;
GRANT USAGE ON SCHEMA config TO facesocial_user;

GRANT CREATE ON SCHEMA core TO facesocial_user;
GRANT CREATE ON SCHEMA face_recognition TO facesocial_user;
GRANT CREATE ON SCHEMA face_detection TO facesocial_user;
GRANT CREATE ON SCHEMA antispoofing TO facesocial_user;
GRANT CREATE ON SCHEMA deepfake_detection TO facesocial_user;
GRANT CREATE ON SCHEMA demographics TO facesocial_user;
GRANT CREATE ON SCHEMA audit TO facesocial_user;
GRANT CREATE ON SCHEMA config TO facesocial_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA face_recognition GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA face_detection GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA antispoofing GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA deepfake_detection GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA demographics GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA config GRANT ALL ON TABLES TO facesocial_user;

-- Set default privileges for sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA face_recognition GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA face_detection GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA antispoofing GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA deepfake_detection GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA demographics GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA config GRANT ALL ON SEQUENCES TO facesocial_user;

-- Create common functions and triggers

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to generate short UUID (for human-readable IDs)
CREATE OR REPLACE FUNCTION generate_short_uuid()
RETURNS TEXT AS $$
BEGIN
    RETURN REPLACE(SUBSTRING(gen_random_uuid()::text FROM 1 FOR 8), '-', '');
END;
$$ LANGUAGE plpgsql;

-- Function for soft delete
CREATE OR REPLACE FUNCTION soft_delete_record()
RETURNS TRIGGER AS $$
BEGIN
    NEW.deleted_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Logging function for audit trail
CREATE OR REPLACE FUNCTION log_table_changes()
RETURNS TRIGGER AS $$
DECLARE
    audit_row audit.table_changes%ROWTYPE;
BEGIN
    audit_row.table_name = TG_TABLE_NAME;
    audit_row.operation = TG_OP;
    audit_row.changed_at = CURRENT_TIMESTAMP;
    audit_row.changed_by = current_user;
    
    IF TG_OP = 'DELETE' THEN
        audit_row.old_values = row_to_json(OLD);
        INSERT INTO audit.table_changes VALUES (audit_row.*);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        audit_row.old_values = row_to_json(OLD);
        audit_row.new_values = row_to_json(NEW);
        INSERT INTO audit.table_changes VALUES (audit_row.*);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        audit_row.new_values = row_to_json(NEW);
        INSERT INTO audit.table_changes VALUES (audit_row.*);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit table for tracking changes
CREATE TABLE IF NOT EXISTS audit.table_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_values JSONB,
    new_values JSONB,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    changed_by TEXT DEFAULT current_user
);

-- Create indexes for audit table
CREATE INDEX IF NOT EXISTS idx_table_changes_table_name ON audit.table_changes(table_name);
CREATE INDEX IF NOT EXISTS idx_table_changes_changed_at ON audit.table_changes(changed_at);
CREATE INDEX IF NOT EXISTS idx_table_changes_operation ON audit.table_changes(operation);

-- Grant permissions on audit table
GRANT ALL ON audit.table_changes TO facesocial_user;

-- Create database info view
CREATE OR REPLACE VIEW public.database_info AS
SELECT
    'FaceSocial AI Database' as database_name,
    version() as postgresql_version,
    current_database() as current_db,
    current_user as current_user,
    CURRENT_TIMESTAMP as created_at,
    '1.0.0' as schema_version;

GRANT SELECT ON public.database_info TO facesocial_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'FaceSocial AI database schemas created successfully!';
    RAISE NOTICE 'Schemas: core, face_recognition, face_detection, antispoofing, deepfake_detection, demographics, audit, config';
    RAISE NOTICE 'Extensions: uuid-ossp, pgcrypto, pg_trgm, btree_gin';
    RAISE NOTICE 'Common functions and audit system ready';
END $$;