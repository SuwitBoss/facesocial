#!/bin/bash

# FaceSocial AI Database Startup Script
# This script sets up and starts all database services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="facesocial-ai"
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p database/init
    mkdir -p database/backup
    mkdir -p config
    mkdir -p logs
    mkdir -p models/{face-detection,face-recognition,antispoofing,deepfake-detection,age-gender}
    
    print_success "Directory structure created"
}

# Create .env file if it doesn't exist
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_status "Creating .env file..."
        
        cat > "$ENV_FILE" << 'EOF'
# FaceSocial AI Database Configuration
POSTGRES_USER=facesocial_user
POSTGRES_PASSWORD=facesocial_2024_secure
POSTGRES_PORT=5432

REDIS_PASSWORD=redis_2024_secure
REDIS_PORT=6379

MILVUS_PORT=19530
MILVUS_HTTP_PORT=9091

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin_2024_secure
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# Development Tools
PGADMIN_EMAIL=admin@facesocial.com
PGADMIN_PASSWORD=admin_2024
PGADMIN_PORT=5050
REDIS_COMMANDER_PORT=8081

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
EOF
        
        print_success ".env file created"
        print_warning "Please review and update the passwords in .env file for production use"
    else
        print_status ".env file already exists"
    fi
}

# Create Milvus configuration
create_milvus_config() {
    print_status "Creating Milvus configuration..."
    
    mkdir -p config
    cat > config/milvus.yaml << 'EOF'
# Milvus configuration for FaceSocial AI
etcd:
  endpoints: 
    - etcd:2379

minio:
  address: minio
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin_2024_secure
  useSSL: false
  bucketName: milvus-bucket

rootPath: files

# Memory settings optimized for face embeddings
queryNode:
  gracefulTime: 1000
  gracefulStopTimeout: 30

# Logging
log:
  level: info
  file:
    rootPath: ""
    maxSize: 300
    maxAge: 10
    maxBackups: 20

# Performance tuning for face recognition workloads
dataNode:
  segment:
    maxSize: 512
    sealProportion: 0.12

indexNode:
  scheduler:
    buildParallel: 1

queryNode:
  segcore:
    chunkRows: 1024
EOF
    
    print_success "Milvus configuration created"
}

# Start database services
start_databases() {
    print_status "Starting database services..."
    print_warning "This may take a few minutes for the first run..."
    
    # Start databases only (no dev tools)
    $DOCKER_COMPOSE up -d postgres redis etcd minio milvus
    
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if $DOCKER_COMPOSE exec -T postgres pg_isready -U facesocial_user -d facesocial &> /dev/null; then
            print_success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "PostgreSQL failed to start within 150 seconds"
            exit 1
        fi
        sleep 5
    done
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    for i in {1..20}; do
        if $DOCKER_COMPOSE exec -T redis redis-cli ping &> /dev/null; then
            print_success "Redis is ready"
            break
        fi
        if [ $i -eq 20 ]; then
            print_error "Redis failed to start within 100 seconds"
            exit 1
        fi
        sleep 5
    done
    
    # Wait for Milvus
    print_status "Waiting for Milvus..."
    for i in {1..40}; do
        if curl -f http://localhost:9091/health &> /dev/null; then
            print_success "Milvus is ready"
            break
        fi
        if [ $i -eq 40 ]; then
            print_error "Milvus failed to start within 200 seconds"
            exit 1
        fi
        sleep 5
    done
    
    print_success "All database services are running!"
}

# Start development tools
start_dev_tools() {
    if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
        print_status "Starting development tools..."
        $DOCKER_COMPOSE --profile dev up -d pgadmin redis-commander
        print_success "Development tools started"
        print_status "Access URLs:"
        echo "  - PgAdmin: http://localhost:5050"
        echo "  - Redis Commander: http://localhost:8081"
        echo "  - MinIO Console: http://localhost:9001"
    fi
}

# Show service status
show_status() {
    print_status "Service Status:"
    $DOCKER_COMPOSE ps
    
    echo ""
    print_status "Connection Information:"
    echo "  PostgreSQL: localhost:5432"
    echo "    Database: facesocial"
    echo "    Username: facesocial_user"
    echo "    Password: (check .env file)"
    echo ""
    echo "  Redis: localhost:6379"
    echo "    Password: (check .env file)"
    echo ""
    echo "  Milvus: localhost:19530"
    echo "    HTTP API: localhost:9091"
    echo ""
    echo "  MinIO: localhost:9000"
    echo "    Console: http://localhost:9001"
    echo "    Access Key: (check .env file)"
}

# Test database connections
test_connections() {
    print_status "Testing database connections..."
    
    # Test PostgreSQL
    if $DOCKER_COMPOSE exec -T postgres psql -U facesocial_user -d facesocial -c "SELECT 'PostgreSQL connection successful' as status;" &> /dev/null; then
        print_success "PostgreSQL connection: OK"
    else
        print_error "PostgreSQL connection: FAILED"
    fi
    
    # Test Redis
    if $DOCKER_COMPOSE exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis connection: OK"
    else
        print_error "Redis connection: FAILED"
    fi
    
    # Test Milvus
    if curl -s http://localhost:9091/health | grep -q "OK"; then
        print_success "Milvus connection: OK"
    else
        print_error "Milvus connection: FAILED"
    fi
}

# Main execution
main() {
    echo "🚀 FaceSocial AI Database Setup"
    echo "=================================="
    
    check_docker
    check_docker_compose
    create_directories
    create_env_file
    create_milvus_config
    start_databases
    start_dev_tools "$1"
    
    echo ""
    show_status
    
    echo ""
    test_connections
    
    echo ""
    print_success "Database setup completed successfully!"
    print_status "You can now start developing AI services that connect to these databases."
    
    if [ "$1" != "--dev" ] && [ "$1" != "-d" ]; then
        echo ""
        print_status "To start development tools, run: $0 --dev"
    fi
    
    echo ""
    print_status "To stop all services: docker compose down"
    print_status "To view logs: docker compose logs -f [service_name]"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --dev, -d    Start with development tools (PgAdmin, Redis Commander)"
        echo "  --help, -h   Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0           Start database services only"
        echo "  $0 --dev     Start databases with development tools"
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac