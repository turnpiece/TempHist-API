#!/usr/bin/env python3
"""
Version bumping script for TempHist API.
Similar to 'npm version patch' but for Python projects.
"""

import re
import sys
import subprocess
from pathlib import Path

def get_current_version():
    """Get current version from version.py"""
    try:
        with open('version.py', 'r') as f:
            content = f.read()
            match = re.search(r'__version__ = "([^"]+)"', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        print("❌ version.py not found")
        sys.exit(1)
    
    print("❌ Could not parse version from version.py")
    sys.exit(1)

def bump_version(current_version, bump_type):
    """Bump version based on type (patch, minor, major)"""
    parts = current_version.split('.')
    if len(parts) != 3:
        print("❌ Invalid version format. Expected: major.minor.patch")
        sys.exit(1)
    
    major, minor, patch = map(int, parts)
    
    if bump_type == 'patch':
        patch += 1
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    else:
        print("❌ Invalid bump type. Use: patch, minor, or major")
        sys.exit(1)
    
    return f"{major}.{minor}.{patch}"

def update_version_file(new_version):
    """Update version.py with new version"""
    try:
        with open('version.py', 'r') as f:
            content = f.read()
        
        # Update __version__
        content = re.sub(
            r'__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
            content
        )
        
        # Update __version_info__
        version_tuple = tuple(map(int, new_version.split('.')))
        content = re.sub(
            r'__version_info__ = \([^)]+\)',
            f'__version_info__ = {version_tuple}',
            content
        )
        
        with open('version.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Updated version.py to {new_version}")
        
    except Exception as e:
        print(f"❌ Error updating version.py: {e}")
        sys.exit(1)

def update_pyproject_toml(new_version):
    """Update pyproject.toml with new version"""
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'version = "[^"]+"',
            f'version = "{new_version}"',
            content
        )
        
        with open('pyproject.toml', 'w') as f:
            f.write(content)
        
        print(f"✅ Updated pyproject.toml to {new_version}")
        
    except FileNotFoundError:
        print("⚠️ pyproject.toml not found, skipping")
    except Exception as e:
        print(f"❌ Error updating pyproject.toml: {e}")

def git_commit_and_tag(new_version):
    """Commit changes and create git tag"""
    try:
        # Add changed files
        subprocess.run(['git', 'add', 'version.py', 'pyproject.toml'], check=True)
        
        # Commit
        subprocess.run(['git', 'commit', '-m', f'Bump version to {new_version}'], check=True)
        print(f"✅ Committed version {new_version}")
        
        # Create tag
        subprocess.run(['git', 'tag', f'v{new_version}'], check=True)
        print(f"✅ Created tag v{new_version}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e}")
        sys.exit(1)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py <patch|minor|major>")
        print("Examples:")
        print("  python bump_version.py patch    # 1.0.0 -> 1.0.1")
        print("  python bump_version.py minor    # 1.0.1 -> 1.1.0")
        print("  python bump_version.py major    # 1.1.0 -> 2.0.0")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"🔄 Bumping version: {current_version} -> {new_version}")
    
    # Update files
    update_version_file(new_version)
    update_pyproject_toml(new_version)
    
    # Git operations
    print("\n📝 Git operations:")
    git_commit_and_tag(new_version)
    
    print(f"\n🎉 Version bump complete!")
    print(f"   Old version: {current_version}")
    print(f"   New version: {new_version}")
    print(f"   Tag created: v{new_version}")
    print(f"\n💡 Next steps:")
    print(f"   git push origin main")
    print(f"   git push origin v{new_version}")

if __name__ == "__main__":
    main()
