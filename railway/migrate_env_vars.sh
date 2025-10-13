#!/bin/bash
# Script to migrate environment variables between Railway environments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Railway Environment Variables Migration${NC}"
echo "=================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found. Please install it first:${NC}"
    echo "npm install -g @railway/cli"
    echo "or visit: https://docs.railway.app/develop/cli"
    exit 1
fi

# Function to copy variables from one project to another
copy_variables() {
    local source_project=$1
    local target_project=$2
    
    echo -e "${YELLOW}📋 Copying variables from ${source_project} to ${target_project}${NC}"
    
    # Get all variables from source project
    echo "Getting variables from source project..."
    railway variables --environment $source_project --json > source_vars.json
    
    if [ ! -s source_vars.json ]; then
        echo -e "${RED}❌ No variables found in source project${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Found variables in source project${NC}"
    
    # Parse and set variables in target project
    echo "Setting variables in target project..."
    
    # Use jq to parse JSON and set variables
    jq -r '.[] | "\(.name)=\(.value)"' source_vars.json | while IFS='=' read -r name value; do
        if [ -n "$name" ] && [ -n "$value" ]; then
            echo "Setting $name..."
            railway variables --set "$name=$value" --environment $target_project
        fi
    done
    
    # Clean up
    rm source_vars.json
    
    echo -e "${GREEN}✅ Variables copied successfully!${NC}"
}

# Main menu
echo "Select migration type:"
echo "1) Copy from develop to staging"
echo "2) Copy from develop to production"
echo "3) Copy from staging to production"
echo "4) Custom source/target"
echo "5) Export variables to file"
echo "6) Import variables from file"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Copying from develop to staging..."
        copy_variables "temphist-api-develop" "temphist-api-staging"
        ;;
    2)
        echo "Copying from develop to production..."
        copy_variables "temphist-api-develop" "temphist-api-production"
        ;;
    3)
        echo "Copying from staging to production..."
        copy_variables "temphist-api-staging" "temphist-api-production"
        ;;
    4)
        read -p "Enter source project name: " source
        read -p "Enter target project name: " target
        copy_variables "$source" "$target"
        ;;
    5)
        read -p "Enter project name to export: " project
        read -p "Enter filename (default: ${project}_vars.json): " filename
        filename=${filename:-${project}_vars.json}
        railway variables --environment $project --json > $filename
        echo -e "${GREEN}✅ Variables exported to $filename${NC}"
        ;;
    6)
        read -p "Enter project name to import to: " project
        read -p "Enter filename to import: " filename
        if [ -f "$filename" ]; then
            jq -r '.[] | "\(.name)=\(.value)"' $filename | while IFS='=' read -r name value; do
                if [ -n "$name" ] && [ -n "$value" ]; then
                    echo "Setting $name..."
                    railway variables --set "$name=$value" --environment $project
                fi
            done
            echo -e "${GREEN}✅ Variables imported successfully!${NC}"
        else
            echo -e "${RED}❌ File $filename not found${NC}"
        fi
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Migration complete!${NC}"
