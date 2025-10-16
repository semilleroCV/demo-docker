#!/bin/bash
#
# Docker Hub Publishing Script for CIFAR-10 Training
# Usage: ./publish_docker.sh [version]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cifar10-training"
DEFAULT_VERSION="latest"

# Get version from argument or use default
VERSION=${1:-$DEFAULT_VERSION}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Docker Hub Publishing Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if DOCKER_USERNAME is set
if [ -z "$DOCKER_USERNAME" ]; then
    echo -e "${YELLOW}DOCKER_USERNAME not set. Please enter your Docker Hub username:${NC}"
    read -p "Username: " DOCKER_USERNAME
    export DOCKER_USERNAME
    echo ""
fi

# Convert username to lowercase for Docker Hub compatibility
DOCKER_USERNAME_LOWER=$(echo "$DOCKER_USERNAME" | tr '[:upper:]' '[:lower:]')
FULL_IMAGE_NAME="${DOCKER_USERNAME_LOWER}/${IMAGE_NAME}"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Image: ${YELLOW}${FULL_IMAGE_NAME}${NC}"
echo -e "  Version: ${YELLOW}${VERSION}${NC}\n"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Step 1: Check Docker login
echo -e "${GREEN}Step 1: Checking Docker Hub authentication...${NC}"
if ! docker info | grep -q "Username: ${DOCKER_USERNAME}"; then
    echo -e "${YELLOW}Not logged in. Attempting to login...${NC}"
    docker login
else
    echo -e "${GREEN}✓ Already logged in as ${DOCKER_USERNAME}${NC}"
fi
echo ""

# Step 2: Build the image
echo -e "${GREEN}Step 2: Building Docker image...${NC}"
docker build -t ${FULL_IMAGE_NAME}:${VERSION} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}\n"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Step 3: Tag as latest if not already
if [ "$VERSION" != "latest" ]; then
    echo -e "${GREEN}Step 3: Tagging as latest...${NC}"
    docker tag ${FULL_IMAGE_NAME}:${VERSION} ${FULL_IMAGE_NAME}:latest
    echo -e "${GREEN}✓ Tagged as latest${NC}\n"
else
    echo -e "${GREEN}Step 3: Skipping additional tagging (already latest)${NC}\n"
fi

# Step 4: Show image info
echo -e "${GREEN}Step 4: Image information:${NC}"
docker images ${FULL_IMAGE_NAME}
echo ""

# Step 5: Test the image (optional)
echo -e "${YELLOW}Would you like to test the image locally before pushing? (y/N)${NC}"
read -p "Test: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Running test...${NC}"
    docker run --rm ${FULL_IMAGE_NAME}:${VERSION} python --version
    echo -e "${GREEN}✓ Test passed${NC}\n"
fi

# Step 6: Push to Docker Hub
echo -e "${YELLOW}Ready to push to Docker Hub. Continue? (y/N)${NC}"
read -p "Push: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Step 5: Pushing to Docker Hub...${NC}"
    
    # Push version tag
    docker push ${FULL_IMAGE_NAME}:${VERSION}
    
    # Push latest tag if version was specified
    if [ "$VERSION" != "latest" ]; then
        docker push ${FULL_IMAGE_NAME}:latest
    fi
    
    echo -e "${GREEN}✓ Push successful${NC}\n"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Publishing Complete!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    echo -e "${GREEN}Your image is now available at:${NC}"
    echo -e "  ${YELLOW}https://hub.docker.com/r/${FULL_IMAGE_NAME}${NC}\n"
    
    echo -e "${GREEN}Others can pull it with:${NC}"
    echo -e "  ${YELLOW}docker pull ${FULL_IMAGE_NAME}:${VERSION}${NC}\n"
    
    echo -e "${GREEN}Run it with:${NC}"
    echo -e "  ${YELLOW}docker run --rm ${FULL_IMAGE_NAME}:${VERSION}${NC}\n"
else
    echo -e "${YELLOW}Push cancelled. Image built locally only.${NC}"
fi

# Cleanup prompt
echo -e "${YELLOW}Would you like to remove local images to free up space? (y/N)${NC}"
read -p "Cleanup: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker image prune -f
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"
