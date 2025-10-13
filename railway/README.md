# Railway Directory

This directory contains all Railway-related deployment and management files for the TempHist API.

## Contents

### Documentation

- **`RAILWAY_DEPLOYMENT.md`** - Complete deployment guide covering:
  - Initial setup and deployment
  - Environment variables configuration
  - Worker service setup
  - Environment migration
  - Troubleshooting
  - Performance optimization

### Scripts

#### Environment Variable Management

- **`railway_export_simple.sh`** - Interactive script for exporting/importing environment variables
- **`export_env_vars.py`** - Python script for advanced environment variable management
- **`migrate_env_vars.sh`** - Batch migration script for multiple environments

#### Diagnostics

- **`check_worker_status.py`** - Diagnostic script to check if worker service is running and processing jobs

## Quick Start

1. **Read the deployment guide:**

   ```bash
   cat railway/RAILWAY_DEPLOYMENT.md
   ```

2. **Check worker status:**

   ```bash
   python railway/check_worker_status.py
   ```

3. **Migrate environment variables:**
   ```bash
   ./railway/railway_export_simple.sh
   ```

## Script Usage

### Environment Variables

```bash
# Interactive script (recommended)
./railway/railway_export_simple.sh

# Python script for advanced usage
python railway/export_env_vars.py

# Batch migration
./railway/migrate_env_vars.sh
```

### Worker Diagnostics

```bash
# Check if worker service is running
python railway/check_worker_status.py
```

## Prerequisites

- Railway CLI installed: `npm install -g @railway/cli`
- Logged into Railway: `railway login`
- `jq` installed for JSON processing (for shell scripts)

## File Organization

All Railway-related files are now organized in this directory instead of scattered in the root. This provides:

- **Cleaner root directory** - Less clutter in main project folder
- **Better organization** - All Railway tools in one place
- **Easier maintenance** - Single location for Railway documentation and scripts
- **Consolidated documentation** - One comprehensive deployment guide instead of multiple files
