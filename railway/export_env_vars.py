#!/usr/bin/env python3
"""
Export and import Railway environment variables using Railway CLI.
This script helps migrate variables between environments.
"""

import subprocess
import json
import sys
import os
from typing import Dict, List

def run_railway_command(command: List[str]) -> Dict:
    """Run a Railway CLI command and return the result."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return {"success": True, "output": result.stdout, "error": None}
    except subprocess.CalledProcessError as e:
        return {"success": False, "output": e.stdout, "error": e.stderr}
    except FileNotFoundError:
        return {"success": False, "output": "", "error": "Railway CLI not found. Install with: npm install -g @railway/cli"}

def export_variables(project_name: str, filename: str = None) -> bool:
    """Export all variables from a Railway project to a JSON file."""
    if not filename:
        filename = f"{project_name}_variables.json"
    
    print(f"📤 Exporting variables from {project_name}...")
    
    # Get variables as JSON
    result = run_railway_command(["railway", "variables", "--environment", project_name, "--json"])
    
    if not result["success"]:
        print(f"❌ Error exporting variables: {result['error']}")
        return False
    
    try:
        variables = json.loads(result["output"])
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(variables, f, indent=2)
        
        print(f"✅ Variables exported to {filename}")
        print(f"📊 Total variables: {len(variables)}")
        
        # Show summary
        for var in variables:
            print(f"  - {var['name']}: {'*' * 8} (hidden)")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing variables JSON: {e}")
        return False

def import_variables(project_name: str, filename: str) -> bool:
    """Import variables from a JSON file to a Railway project."""
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found")
        return False
    
    print(f"📥 Importing variables to {project_name}...")
    
    try:
        with open(filename, 'r') as f:
            variables = json.load(f)
        
        success_count = 0
        error_count = 0
        
        for var in variables:
            name = var.get('name')
            value = var.get('value')
            
            if not name or not value:
                print(f"⚠️  Skipping invalid variable: {var}")
                error_count += 1
                continue
            
            # Set variable
            result = run_railway_command(["railway", "variables", "--set", f"{name}={value}", "--environment", project_name])
            
            if result["success"]:
                print(f"  ✅ Set {name}")
                success_count += 1
            else:
                print(f"  ❌ Failed to set {name}: {result['error']}")
                error_count += 1
        
        print(f"\n📊 Import Summary:")
        print(f"  ✅ Successfully imported: {success_count}")
        print(f"  ❌ Failed to import: {error_count}")
        
        return error_count == 0
        
    except json.JSONDecodeError as e:
        print(f"❌ Error reading variables file: {e}")
        return False

def list_projects() -> List[str]:
    """List all Railway environments."""
    result = run_railway_command(["railway", "environments"])
    
    if not result["success"]:
        print(f"❌ Error listing projects: {result['error']}")
        return []
    
    # Parse project names from output
    lines = result["output"].strip().split('\n')
    projects = []
    
    for line in lines[1:]:  # Skip header
        if line.strip():
            # Extract project name (first column)
            project_name = line.split()[0]
            projects.append(project_name)
    
    return projects

def main():
    """Main function with interactive menu."""
    print("🚀 Railway Environment Variables Manager")
    print("=" * 50)
    
    # Check if Railway CLI is available
    result = run_railway_command(["railway", "--version"])
    if not result["success"]:
        print("❌ Railway CLI not found!")
        print("Install it with: npm install -g @railway/cli")
        return
    
    print("✅ Railway CLI found")
    
    while True:
        print("\nSelect an option:")
        print("1) Export variables from project")
        print("2) Import variables to project")
        print("3) List all environments")
        print("4) Export from develop → staging")
        print("5) Export from develop → production")
        print("6) Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            project = input("Enter project name: ").strip()
            filename = input("Enter filename (or press Enter for default): ").strip()
            if not filename:
                filename = f"{project}_variables.json"
            export_variables(project, filename)
            
        elif choice == "2":
            project = input("Enter project name: ").strip()
            filename = input("Enter filename: ").strip()
            import_variables(project, filename)
            
        elif choice == "3":
            projects = list_projects()
            if projects:
                print("\n📋 Available environments:")
                for project in projects:
                    print(f"  - {project}")
            else:
                print("❌ No environments found")
                
        elif choice == "4":
            export_variables("temphist-api-develop", "develop_variables.json")
            print("\n📝 Next steps:")
            print("1. Review develop_variables.json")
            print("2. Adjust staging-specific values")
            print("3. Run: python export_env_vars.py")
            print("4. Choose option 2 to import to staging")
            
        elif choice == "5":
            export_variables("temphist-api-develop", "develop_variables.json")
            print("\n📝 Next steps:")
            print("1. Review develop_variables.json")
            print("2. Adjust production-specific values")
            print("3. Run: python export_env_vars.py")
            print("4. Choose option 2 to import to production")
            
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
