#!/bin/bash
# Simple Railway environment variables export/import script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Railway Environment Variables Manager${NC}"
echo "=================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found. Please install it first:${NC}"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged in to Railway. Please run: railway login${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Railway CLI found and logged in${NC}"

# Function to export variables from current environment
export_current_env() {
    local filename=$1
    
    echo -e "${BLUE}📤 Exporting variables from current environment...${NC}"
    
    # Get current project info
    local project_info=$(railway status --json 2>/dev/null)
    if [ -z "$project_info" ]; then
        echo -e "${RED}❌ Not connected to a Railway project${NC}"
        echo "Please run: railway link"
        return 1
    fi
    
    # Export variables
    railway variables --json > "$filename"
    
    if [ -s "$filename" ]; then
        echo -e "${GREEN}✅ Variables exported to $filename${NC}"
        
        # Show summary
        local count=$(jq length "$filename" 2>/dev/null || echo "0")
        echo -e "${BLUE}📊 Total variables: $count${NC}"
        
        # Show variable names (without values for security)
        echo -e "${BLUE}📋 Variable names:${NC}"
        jq -r '.[].name' "$filename" 2>/dev/null | while read -r name; do
            echo "  - $name"
        done
    else
        echo -e "${RED}❌ No variables found or export failed${NC}"
        return 1
    fi
}

# Function to import variables to current environment
import_to_current_env() {
    local filename=$1
    
    if [ ! -f "$filename" ]; then
        echo -e "${RED}❌ File $filename not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📥 Importing variables to current environment...${NC}"
    
    # Check if file has valid JSON
    if ! jq empty "$filename" 2>/dev/null; then
        echo -e "${RED}❌ Invalid JSON file${NC}"
        return 1
    fi
    
    local success_count=0
    local error_count=0
    
    # Import each variable
    jq -r '.[] | "\(.name)=\(.value)"' "$filename" | while IFS='=' read -r name value; do
        if [ -n "$name" ] && [ -n "$value" ]; then
            echo "Setting $name..."
            if railway variables --set "$name=$value" >/dev/null 2>&1; then
                echo "  ✅ $name"
                ((success_count++))
            else
                echo "  ❌ Failed to set $name"
                ((error_count++))
            fi
        fi
    done
    
    echo -e "${GREEN}📊 Import Summary:${NC}"
    echo "  ✅ Successfully imported: $success_count"
    echo "  ❌ Failed to import: $error_count"
}

# Function to show current project status
show_status() {
    echo -e "${BLUE}📋 Current Railway Status:${NC}"
    railway status
    echo ""
    echo -e "${BLUE}📋 Current Variables:${NC}"
    railway variables --kv
}

# Main menu
while true; do
    echo ""
    echo -e "${YELLOW}Select an option:${NC}"
    echo "1) Show current status and variables"
    echo "2) Export variables from current environment"
    echo "3) Import variables to current environment"
    echo "4) Export from develop environment"
    echo "5) Export from staging environment"
    echo "6) Export from production environment"
    echo "7) Exit"
    
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            show_status
            ;;
        2)
            read -p "Enter filename (default: current_env_vars.json): " filename
            filename=${filename:-current_env_vars.json}
            export_current_env "$filename"
            ;;
        3)
            read -p "Enter filename to import: " filename
            import_to_current_env "$filename"
            ;;
        4)
            echo -e "${YELLOW}📝 Instructions for exporting from develop:${NC}"
            echo "1. Run: railway link"
            echo "2. Select your develop project"
            echo "3. Run this script again and choose option 2"
            echo "4. Save the file as develop_vars.json"
            ;;
        5)
            echo -e "${YELLOW}📝 Instructions for exporting from staging:${NC}"
            echo "1. Run: railway link"
            echo "2. Select your staging project"
            echo "3. Run this script again and choose option 2"
            echo "4. Save the file as staging_vars.json"
            ;;
        6)
            echo -e "${YELLOW}📝 Instructions for exporting from production:${NC}"
            echo "1. Run: railway link"
            echo "2. Select your production project"
            echo "3. Run this script again and choose option 2"
            echo "4. Save the file as production_vars.json"
            ;;
        7)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            break
            ;;
        *)
            echo -e "${RED}❌ Invalid choice${NC}"
            ;;
    esac
done
