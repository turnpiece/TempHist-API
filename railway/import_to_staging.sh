#!/bin/bash
# Script to import develop variables to staging with appropriate adjustments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Import Develop Variables to Staging${NC}"
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

# Check if develop_env_vars.json exists
if [ ! -f "develop_env_vars.json" ]; then
    echo -e "${RED}❌ develop_env_vars.json not found${NC}"
    echo "Please export variables from develop first using:"
    echo "  railway link  # Link to develop project"
    echo "  railway variables --json > develop_env_vars.json"
    exit 1
fi

echo -e "${BLUE}📋 Current Railway Status:${NC}"
railway status

echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Make sure you're linked to your STAGING project${NC}"
echo "If you're not sure, run: railway link"
echo ""

read -p "Press Enter to continue with import, or Ctrl+C to cancel..."

echo -e "${BLUE}📥 Importing variables to staging environment...${NC}"

# Import all variables from develop
success_count=0
error_count=0

jq -r 'to_entries[] | "\(.key)=\(.value)"' develop_env_vars.json | while IFS='=' read -r name value; do
    # Skip Railway-specific variables that shouldn't be copied
    if [[ "$name" =~ ^RAILWAY_ ]]; then
        echo "  ⏭️  Skipping Railway-specific variable: $name"
        continue
    fi
    
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

echo ""
echo -e "${BLUE}🔧 Applying staging-specific adjustments...${NC}"

# Apply staging-specific settings
staging_adjustments=(
    "DEBUG=false"
    "CACHE_WARMING_ENABLED=false"
    "CACHE_WARMING_CONCURRENT_REQUESTS=2"
    "CACHE_WARMING_MAX_LOCATIONS=10"
)

for adjustment in "${staging_adjustments[@]}"; do
    name=$(echo "$adjustment" | cut -d'=' -f1)
    value=$(echo "$adjustment" | cut -d'=' -f2)
    
    echo "Setting $name=$value (staging override)..."
    if railway variables --set "$name=$value" >/dev/null 2>&1; then
        echo "  ✅ $name"
    else
        echo "  ❌ Failed to set $name"
    fi
done

echo ""
echo -e "${GREEN}📊 Import Summary:${NC}"
echo "  ✅ Variables imported from develop"
echo "  ✅ Staging-specific adjustments applied"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Verify variables in Railway dashboard"
echo "2. Deploy your staging environment"
echo "3. Test the staging deployment"
echo "4. Check worker service is running"
echo ""
echo -e "${YELLOW}⚠️  Remember to update these for staging:${NC}"
echo "  - API keys (if using different keys for staging)"
echo "  - Redis URL (if using separate Redis instance)"
echo "  - Domain URLs (if using different domains)"
