# TFHE Library - Custom Modifications for HDM_rs

## Overview
This is a forked version of the TFHE-rs library with custom modifications to support the HDM_rs project.

## Custom Modifications

The following modifications have been made to support homomorphic matching operations:

### 1. **client_key/mod.rs** (Shortint & Boolean)
- Added `encrypt_abs()` method to encrypt raw u32 values (absolute plaintext values) in `tfhe/src/shortint/client_key/mod.rs` and `tfhe/src/boolean/client_key/mod.rs`.
- Added `glwe_secret_key()` method to `tfhe/src/shortint/client_key/mod.rs` to expose the GLWE secret key view.

### 2. **engine/bootstrapping.rs**
- Made `Bootstrapper` struct public
- Made `apply_bootstrapping_pattern()` method public

### 3. **engine/mod.rs**
- Made `BinaryGatesEngine` trait public
- Made `encryption_generator` field public
- Made `bootstrapper` field public
- Added `encrypt_abs()` method

### 4. **parameters/mod.rs**
- Modified `DEFAULT_PARAMETERS` (Note: These changes were adopted by upstream)

## Viewing Your Modifications

### See modified commits:
```bash
git log --oneline main ^origin/main
```

### See modified files:
```bash
git diff origin/main --stat
```

### See detailed changes:
```bash
git diff origin/main
```

### Create a patch file:
```bash
git diff origin/main > my_modifications.patch
```

## Updating TFHE Library

### Automated Update (Recommended)
```bash
./UPDATE_TFHE.sh
```

This script will:
1. Show your current modifications
2. Fetch latest updates from upstream
3. Create automatic backups
4. Apply your modifications on top of latest code
5. Handle conflicts (if any)

### Manual Update Process

If you prefer to update manually or need to handle complex conflicts:

```bash
# 1. Check current state
git status
git log --oneline main ^origin/main

# 2. Create backup
git branch backup-$(date +%Y%m%d) main

# 3. Fetch updates
git fetch origin

# 4. See what's new
git log --oneline main..origin/main | head -20

# 5. Create update branch
git checkout -b update-$(date +%Y%m%d) origin/main

# 6. Apply your modifications
git cherry-pick main

# 7. If conflicts occur, resolve them:
#    - Edit conflicted files
#    - git add <resolved-files>
#    - git cherry-pick --continue

# 8. Update main branch
git checkout main
git reset --hard update-$(date +%Y%m%d)

# 9. Test with HDM_rs
cd ../HDM_rs
cargo build --release --bin cloud_odm
cargo run --release --bin cloud_odm
```

## Important Branches

- `main` - Your working branch (upstream + your modifications)
- `my-modifications-backup` - Original backup of your modifications
- `origin/main` - Upstream TFHE repository (read-only)

## Testing After Update

After updating, always test with your HDM_rs project:

```bash
cd ../HDM_rs
cargo clean
cargo build --release --bin cloud_odm
cargo run --release --bin cloud_odm
```

## Troubleshooting

### Conflict Resolution Tips

1. **Type changes (e.g., ActivatedRandomGenerator → DefaultRandomGenerator)**
   - Use the upstream's type (they know their internals best)
   - Keep your visibility changes (pub)

2. **Parameter changes**
   - If upstream adopted similar values, use theirs
   - Otherwise, carefully evaluate if your values still make sense

3. **Method signature changes**
   - Adapt your code to new signatures
   - Keep the core functionality you need

### Rust Version Requirements

The latest TFHE requires Rust 1.93.0 or higher:

```bash
rustup update
rustc --version  # Should be 1.93.0 or higher
```

### Cargo.toml in HDM_rs

Make sure your `HDM_rs/Cargo.toml` has:
```toml
[dependencies]
tfhe = { path = "../tfhe-rs/tfhe", features = [ "boolean" ] }
```

Note: `x86_64-unix` feature was removed in newer versions.

## Backup Strategy

Every update automatically creates:
1. A timestamped backup branch
2. A patch file in `/tmp/my_tfhe_modifications_YYYYMMDD_HHMMSS.patch`

You can restore from backup:
```bash
git checkout main
git reset --hard backup-YYYYMMDD
```

## Last Updated
- Date: February 10, 2026
- TFHE Commit: 2f37197b3
- Rust Version: 1.93.0

## Contact
For questions about these modifications, contact the HDM_rs project maintainers.
