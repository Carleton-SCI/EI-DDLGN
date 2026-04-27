#!/bin/bash
# Script to update tfhe-rs subtree from upstream
# This script can be run from anywhere in the repo.

set -e

# Resolve project root
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

echo "=========================================="
echo "TFHE Subtree Update Script"
echo "=========================================="

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  WARNING: You have uncommitted changes."
    echo "It is recommended to commit or stash them before updating the subtree."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Check/Add remote
if ! git remote | grep -q "^tfhe-upstream$"; then
    echo "Adding remote 'tfhe-upstream'..."
    git remote add tfhe-upstream https://github.com/zama-ai/tfhe-rs.git
else
    echo "Remote 'tfhe-upstream' found."
fi

# Fetch
echo "Fetching latest changes from upstream..."
git fetch tfhe-upstream

# --- SAFETY NET START ---
# Create a backup timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BRANCH="tfhe-backup-$TIMESTAMP"
PATCH_FILE="/tmp/tfhe_modifications_$TIMESTAMP.patch"

echo "Creating backup branch '$BACKUP_BRANCH'..."
git branch "$BACKUP_BRANCH"

# List of files we KNOW are ours and should never be deleted
PROTECTED_FILES=(
    "tfhe-rs/CUSTOM_MODIFICATIONS.md"
    "tfhe-rs/QUICK_REFERENCE.md"
    "tfhe-rs/MAINTAIN_TFHE.md"
    "tfhe-rs/update_tfhe.sh"
)

echo "Saving your custom modifications to '$PATCH_FILE'..."
# We create a patch of changes between upstream and your version
# This captures your "internal code modifications"
git diff tfhe-upstream/main HEAD:tfhe-rs > "$PATCH_FILE"
# --- SAFETY NET END ---

# Pull
echo "Merging upstream changes into tfhe-rs folder..."
echo "Running: git subtree pull --prefix=tfhe-rs tfhe-upstream main --squash"
git subtree pull --prefix=tfhe-rs tfhe-upstream main --squash || {
    echo "❌ Subtree pull failed. Restoring from backup..."
    git reset --hard "$BACKUP_BRANCH"
    exit 1
}

# --- RESTORATION START ---
echo "Checking for lost custom files..."
for file in "${PROTECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Restoring '$file' from backup..."
        git checkout "$BACKUP_BRANCH" -- "$file"
    fi
done

echo "Ensuring your internal code modifications are present..."
# We try to apply the patch we saved. 
# --3way allows git to merge intelligently if upstream also changed the file.
# If the patch is empty (no custom changes), this does nothing.
if [ -s "$PATCH_FILE" ]; then
    echo "Re-applying your local modifications..."
    # We must be in the tfhe-rs directory for the patch to apply correctly 
    # because the patch was generated relative to that root (HEAD:tfhe-rs)
    cd tfhe-rs
    if git apply --3way "$PATCH_FILE"; then
        echo "✅ Your custom changes were re-applied successfully."
    else
        echo "⚠️  CONFLICTS DETECTED during re-application of your changes."
        echo "Git has applied what it could, but some changes conflicted with upstream updates."
        echo "Please check 'git status' and resolve the conflicts manually."
        echo "Your original code is safely available in branch: $BACKUP_BRANCH"
    fi
    cd ..
else
    echo "No custom modifications detected to re-apply."
fi

# Check for other modified files that might have been overwritten
echo ""
echo "Comparing final result with your backup..."
echo "Files different from your backup (showing Upstream's new work + your kept changes):"
git diff --stat "$BACKUP_BRANCH" -- tfhe-rs/
# --- RESTORATION END ---

echo "=========================================="
echo "✅ Update complete!"
echo "Please verify the changes and resolve any conflicts if they occurred."
echo "=========================================="
