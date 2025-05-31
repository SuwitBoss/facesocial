-- Create FaceSocial AI Database User
-- This script creates the application user and sets up basic permissions
-- Run as superuser (postgres)

-- Create application user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facesocial_user') THEN
        CREATE USER facesocial_user WITH 
            PASSWORD 'facesocial_2024_secure'
            CREATEDB
            LOGIN;
        RAISE NOTICE 'Created user: facesocial_user';
    ELSE
        RAISE NOTICE 'User facesocial_user already exists';
    END IF;
END
$$;

-- Create application database if it doesn't exist
SELECT 'CREATE DATABASE facesocial OWNER facesocial_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'facesocial')\gexec

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE facesocial TO facesocial_user;
GRANT CONNECT ON DATABASE facesocial TO facesocial_user;

-- Switch to facesocial database for schema operations
\c facesocial

-- Grant schema creation permissions
GRANT CREATE ON DATABASE facesocial TO facesocial_user;
GRANT USAGE ON SCHEMA public TO facesocial_user;
GRANT CREATE ON SCHEMA public TO facesocial_user;

-- Grant permissions for sequences and tables
ALTER DEFAULT PRIVILEGES FOR ROLE facesocial_user GRANT ALL ON TABLES TO facesocial_user;
ALTER DEFAULT PRIVILEGES FOR ROLE facesocial_user GRANT ALL ON SEQUENCES TO facesocial_user;
ALTER DEFAULT PRIVILEGES FOR ROLE facesocial_user GRANT EXECUTE ON FUNCTIONS TO facesocial_user;

-- Create read-only user for monitoring and analytics
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facesocial_readonly') THEN
        CREATE USER facesocial_readonly WITH 
            PASSWORD 'facesocial_readonly_2024'
            LOGIN;
        RAISE NOTICE 'Created read-only user: facesocial_readonly';
    ELSE
        RAISE NOTICE 'User facesocial_readonly already exists';
    END IF;
END
$$;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE facesocial TO facesocial_readonly;
GRANT USAGE ON SCHEMA public TO facesocial_readonly;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database users created successfully!';
    RAISE NOTICE 'Application user: facesocial_user (full access)';
    RAISE NOTICE 'Read-only user: facesocial_readonly (monitoring)';
    RAISE NOTICE 'Database: facesocial';
END $$;